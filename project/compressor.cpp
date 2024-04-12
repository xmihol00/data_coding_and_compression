#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint32_t width)
    : _model(model), _adaptive(adaptive), _width(width) { }

Compressor::~Compressor()
{ 
    if (_image != nullptr)
    {
        free(_image);
        _image = nullptr;
    }
}

void Compressor::computeHistogram()
{
    for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        _histogram[i].pixelValue = i;
    }

    for (uint32_t i = 0; i < _size; i++)
    {
        _histogram[_image[i]].count++;
    }
}

void Compressor::fixSortedNodes()
{
    uint16_t idx = _sortedNodesTail - 1;
    Node tail = _sortedNodes[_sortedNodesTail];
    while (idx >= _sortedNodesHead && _sortedNodes[idx].count > tail.count)
    {
        cerr << "sorting " << idx << " " << _sortedNodesHead << " " << _sortedNodes[idx].count << " " << tail.count << endl;   
        _sortedNodes[idx + 1] = _sortedNodes[idx];
        idx--;
    }
    _sortedNodes[idx + 1] = tail;
}

void Compressor::populateCodeTable()
{
    v8_int32 lastCode = _mm256_set1_epi32(0);
    uint32_t delta = 0;
    queue<Node> nodeQueue;
    _tree[_treeIndex].count = 0; // use the count as depth counter
    nodeQueue.push(_tree[_treeIndex]);
    
    while (nodeQueue.front().isNode())
    {
        Node node = nodeQueue.front();
        _tree[node.left].count = node.count + 1;
        _tree[node.right].count = node.count + 1;
        nodeQueue.push(_tree[node.left]);
        nodeQueue.push(_tree[node.right]);
        nodeQueue.pop();
    }

    uint32_t lastDepth = nodeQueue.front().count;
    _codeTable[nodeQueue.front().right] = lastCode;
    nodeQueue.pop();

    while (!nodeQueue.empty())
    {
        Node node = nodeQueue.front();
        if (node.isLeaf())
        {

            delta = node.count - lastDepth;
            cerr << "Delta: " << delta << ", Last depth: " << lastDepth << ", Current depth: " << node.count << endl;
            lastDepth = node.count;
            
            v8_int32 incremented = _mm256_add_epi32(lastCode, VINT8_LSB);
            v8_int32 leftShifted = _mm256_sll_epi32(incremented, _mm_set_epi32(0, 0, 0, delta));
            v8_int32 rightShifted = _mm256_srl_epi32(incremented, _mm_set_epi32(0, 0, 0, 32 - delta));
            v8_int32 permuted = _mm256_permutevar8x32_epi32(rightShifted, _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1));
            v8_int32 masked = _mm256_and_si256(permuted, _mm256_set_epi32(0x00000000, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff));
            lastCode = _mm256_or_si256(masked, leftShifted);

            _codeTable[node.right] = lastCode;
        }
        else
        {
            _tree[node.left].count = node.count + 1;
            _tree[node.right].count = node.count + 1;
            nodeQueue.push(_tree[node.left]);
            nodeQueue.push(_tree[node.right]);
        }
        nodeQueue.pop();
    }
}

void Compressor::printTree(uint16_t nodeIdx, uint16_t indent = 0)
{
    if (_tree[nodeIdx].isLeaf())
    {
        Leaf leaf = reinterpret_cast<Leaf *>(_tree)[nodeIdx];
        cerr << string(indent, ' ') <<  nodeIdx << ": Leaf: " << leaf.count << " " << (char)(leaf.pixelValue) << endl;
    }
    else
    {
        cerr << string(indent, ' ') <<  nodeIdx << ": Node: " << _tree[nodeIdx].count << " " << _tree[nodeIdx].left << " " << _tree[nodeIdx].right << endl;
        printTree(_tree[nodeIdx].left, indent + 2);
        printTree(_tree[nodeIdx].right, indent + 2);
    }
}

void Compressor::compress(string inputFileName, string outputFileName)
{
    readInputFile(inputFileName);
    compressStatic();
}

void Compressor::readInputFile(std::string inputFileName)
{
    ifstream inputFile(inputFileName, ios::binary);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file '" << inputFileName << "'." << endl;
        exit(1);
    }

    // get the size of the input file
    inputFile.seekg(0, ios::end);
    _size = inputFile.tellg();
    inputFile.seekg(0, ios::beg);

    _height = _size / _width;
    if (_height * _width != _size) // verify that file is rectangular
    {
        cerr << "Error: The input file size is not a multiple of the specified width." << endl;
        exit(1);
    }

    _image = static_cast<symbol_t *>(aligned_alloc(64, _width * _height * sizeof(symbol_t)));
    if (_image == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the image." << endl;
        exit(1);
    }

    inputFile.read(reinterpret_cast<char *>(_image), _size);
}

void Compressor::compressStatic()
{
    computeHistogram();

    // sort the histogram by frequency in ascending order
    sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
    
    // create the Huffman tree starting from first character that occurred at least once
    uint16_t histogramIndex = 0;
    while (_histogram[histogramIndex].count == 0) // filter out characters that did not occur
    {
        histogramIndex++;
    }

    // pointers to the same memory
    Leaf *leafTree = reinterpret_cast<Leaf *>(_tree);
    Node *nodeTree = _tree;

    leafTree[1] = _histogram[histogramIndex++]; // TODO: solve only one character in histogram
    leafTree[2] = _histogram[histogramIndex++];
    nodeTree[3].count = leafTree[1].count + leafTree[2].count;
    nodeTree[3].left = 1;
    nodeTree[3].right = 2;
    _treeIndex = 3;

    _sortedNodesHead = NUMBER_OF_SYMBOLS << 1; // TODO: solve issue with underflow
    _sortedNodesTail = _sortedNodesHead;
    _sortedNodes[_sortedNodesHead] = nodeTree[3];
    _sortedNodes[_sortedNodesHead].left = 3;

    bool connectTree = false;

    while (histogramIndex < NUMBER_OF_SYMBOLS)
    {
        if (histogramIndex + 1 < NUMBER_OF_SYMBOLS && _histogram[histogramIndex + 1].count < _sortedNodes[_sortedNodesHead].count) // node from 2 symbols
        {
            leafTree[++_treeIndex] = _histogram[histogramIndex++];
            leafTree[++_treeIndex] = _histogram[histogramIndex++];
            
            ++_treeIndex;
            nodeTree[_treeIndex].count = leafTree[_treeIndex - 2].count + leafTree[_treeIndex - 1].count;
            nodeTree[_treeIndex].left = _treeIndex - 2;
            nodeTree[_treeIndex].right = _treeIndex - 1;

            connectTree = _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count;
            cerr << "double connect " << _treeIndex << " " << connectTree << endl;
        }
        else // node from 1 symbol and already existing sub-tree
        {
            leafTree[++_treeIndex] = _histogram[histogramIndex++];
            connectTree = true;
            cerr << "single connect " << _treeIndex << " " << connectTree << endl;
        }

        if (connectTree) // newly created node must be connected to the existing tree
        {
            do
            {
                nodeTree[_treeIndex + 1].count = _sortedNodes[_sortedNodesHead].count + nodeTree[_treeIndex].count;
                nodeTree[_treeIndex + 1].left = _sortedNodes[_sortedNodesHead++].left;
                nodeTree[_treeIndex + 1].right = _treeIndex;
                _treeIndex++;
                cerr << "connect " << _treeIndex << " " << _sortedNodesHead << " " << _sortedNodesTail << endl;
            }
            while (_sortedNodesHead <= _sortedNodesTail && _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count);
            
            if (_sortedNodesHead >= _sortedNodesTail)
            {
                cerr << _sortedNodesHead << " = " << _sortedNodesTail << endl;
            }
            _sortedNodes[++_sortedNodesTail] = nodeTree[_treeIndex];
            _sortedNodes[_sortedNodesTail].left = _treeIndex;
            fixSortedNodes();
            cerr << "connected " << _treeIndex << " " << _sortedNodesHead << " " << _sortedNodesTail << endl;
        }
        else // newly created node has the smallest count
        {
            //cerr << "no connect " << _treeIndex << " " << _sortedNodesHead << " " << _sortedNodesTail << endl;
            _sortedNodes[--_sortedNodesHead] = nodeTree[_treeIndex];
            _sortedNodes[_sortedNodesHead].left = _treeIndex;
            //cerr << "no connected " << _treeIndex << " " << _sortedNodesHead << " " << _sortedNodesTail << endl;
        }
    }

    if (_sortedNodesHead < _sortedNodesTail)
    {
        cerr << _sortedNodes[_sortedNodesHead].count << " - fst\n";
        _sortedNodesHead++;
        for (uint16_t i = _sortedNodesHead; i <= _sortedNodesTail; i++)
        {
            cerr << _sortedNodes[i].count << " " << _treeIndex << " " << _sortedNodes[i].left << "\n";
            nodeTree[_treeIndex + 1].count = _sortedNodes[i].count + nodeTree[_treeIndex].count;
            nodeTree[_treeIndex + 1].left = _sortedNodes[i].left;
            nodeTree[_treeIndex + 1].right = _treeIndex;
            _treeIndex++;
        }
        cerr << _sortedNodesHead << " < " << _sortedNodesTail << " " << _sortedNodes[_sortedNodesHead].left << " " << _sortedNodes[_sortedNodesTail].left << endl;
    }

    /*if (histogramIndex < NUMBER_OF_SYMBOLS)
    {
        leafTree[++_treeIndex] = _histogram[histogramIndex];
        nodeTree[_treeIndex + 1].count = leafTree[_treeIndex].count + _sortedNodes[_sortedNodesHead].count;
        nodeTree[_treeIndex + 1].left = _treeIndex;
        nodeTree[_treeIndex + 1].right = _sortedNodes[_sortedNodesHead].left;
        _treeIndex++;
    }*/

    printTree(_treeIndex);

    for (uint16_t i = 1; i <= _treeIndex; i++)
    {
        if (_tree[i].isLeaf())
        {
            Leaf leaf = reinterpret_cast<Leaf *>(_tree)[i];
            cerr << i << ": Leaf: " << leaf.count << " " << (char)leaf.pixelValue << endl;
        }
        else
        {
            cerr << i << ": Node: " << _tree[i].count << " " << _tree[i].left << " " << _tree[i].right << endl;
        }
    }

    populateCodeTable();

    symbol_t seen[NUMBER_OF_SYMBOLS] = {0};
    for (size_t i = 0; i < _size; i++)
    {
        if (seen[_image[i]] == 0)
        {
            seen[_image[i]] = 1;
            #if __AVX512CD__ && __AVX512VL__
                uint32_t code[8];
                _mm256_storeu_si256((__m256i *)code, _codeTable[_image[i]]);
                v4_int64 leadingZeros = _mm256_lzcnt_epi32(_codeTable[_image[i]]);
                uint32_t leadingZerosAlias[8];
                _mm256_storeu_si256((__m256i *)leadingZerosAlias, leadingZeros);

                cerr << (char)_image[i] << ": ";
                int j = 0;
                for (; j < 8; j++)
                {
                    if (leadingZerosAlias[j] < 32)
                    {
                        bitset<32> codeBitset(code[j]);
                        for (int k = 31 - leadingZerosAlias[j]; k >= 0; k--)
                        {
                            cerr << codeBitset[k];
                        }
                        j++;
                        break;
                    }
                }
                for (; j < 8; j++)
                {
                    bitset<32> codeBitset(code[j]);
                    cerr << codeBitset;
                }
                if (leadingZerosAlias[7] == 32)
                {
                    cerr << "0";
                }
                cerr << endl;
            #else
                uint32_t code[8];
                bitset<64> codeBitset(code[0]);
                _mm256_storeu_si256((__m256i *)code, _codeTable[_image[i]]);
                cerr << (void *)_image[i] << ": ";
                for (int j = 0; j < 8; j++)
                {
                    bitset<32> codeBitset(code[j]);
                    cerr << codeBitset;
                }
                cerr << endl;    
            #endif
        }
    }
}

