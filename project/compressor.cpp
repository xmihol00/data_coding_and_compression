#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint64_t width)
    : _model(model), _adaptive(adaptive), _width(width) { }

Compressor::~Compressor()
{ 
    if (_dataPool != nullptr)
    {
        free(_dataPool);
    }
}

void Compressor::computeHistogram()
{
    for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        _histogram[i].symbol = i;
    }

    for (uint64_t i = 0; i < _size; i++)
    {
        _histogram[_fileData[i]].count++;
    }
}

void Compressor::buildHuffmanTree()
{
    // create the Huffman tree starting from first character that occurred at least once
    uint16_t histogramIndex = 0;
    while (_histogram[histogramIndex].count == 0) // filter out characters that did not occur
    {
        histogramIndex++;
    }

    // pointers to the same memory
    Leaf *leafTree = reinterpret_cast<Leaf *>(_tree);
    Node *nodeTree = _tree;

    if (histogramIndex == NUMBER_OF_SYMBOLS - 1) // only a single type of symbol
    {
        leafTree[1] = _histogram[histogramIndex];
        _treeIndex = 1;
    }
    else
    {
        leafTree[1] = _histogram[histogramIndex++];
        leafTree[2] = _histogram[histogramIndex++];
        nodeTree[3].count = leafTree[1].count + leafTree[2].count;
        nodeTree[3].left = 1;
        nodeTree[3].right = 2;
        _treeIndex = 3;

        _sortedNodesHead = NUMBER_OF_SYMBOLS - 1;
        _sortedNodesTail = NUMBER_OF_SYMBOLS - 1;
        _sortedNodes[NUMBER_OF_SYMBOLS - 1] = nodeTree[3];
        _sortedNodes[NUMBER_OF_SYMBOLS - 1].left = 3;

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
            }
            else // node from 1 symbol and already existing sub-tree
            {
                leafTree[++_treeIndex] = _histogram[histogramIndex++];
                connectTree = true;
            }

            if (connectTree) // newly created node must be connected to the existing tree
            {
                do
                {
                    nodeTree[_treeIndex + 1].count = _sortedNodes[_sortedNodesHead].count + nodeTree[_treeIndex].count;
                    nodeTree[_treeIndex + 1].left = _sortedNodes[_sortedNodesHead++].left;
                    nodeTree[_treeIndex + 1].right = _treeIndex;
                    _treeIndex++;
                } while (_sortedNodesHead <= _sortedNodesTail && _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count);

                _sortedNodes[++_sortedNodesTail] = nodeTree[_treeIndex];
                _sortedNodes[_sortedNodesTail].left = _treeIndex;

                // buble sort the new node into the sorted nodes
                uint16_t idx = _sortedNodesTail - 1;
                Node tail = _sortedNodes[_sortedNodesTail];
                while (idx >= _sortedNodesHead && _sortedNodes[idx].count > tail.count)
                {
                    _sortedNodes[idx + 1] = _sortedNodes[idx];
                    idx--;
                }
                _sortedNodes[idx + 1] = tail;
            }
            else // newly created node has the smallest count
            {
                _sortedNodes[--_sortedNodesHead] = nodeTree[_treeIndex];
                _sortedNodes[_sortedNodesHead].left = _treeIndex;
            }
        }

        for (uint16_t i = ++_sortedNodesHead; i <= _sortedNodesTail; i++)
        {
            nodeTree[_treeIndex + 1].count = _sortedNodes[i].count + nodeTree[_treeIndex].count;
            nodeTree[_treeIndex + 1].left = _sortedNodes[i].left;
            nodeTree[_treeIndex + 1].right = _treeIndex;
            _treeIndex++;
        }
    }
}

void Compressor::populateCodeTable()
{
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++) // clear the code table, TODO: possibly useless
    {
        _codeTable[i] = 0;
    }

    uint32_t lastCode = 0;
    uint32_t delta = 0;
    queue<Node> nodeQueue;       // TODO optimize with array
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

    uint32_t lastDepth = 0;
    while (!nodeQueue.empty())
    {
        Node node = nodeQueue.front();
        if (node.isLeaf())
        {

            delta = node.count - lastDepth;
            lastDepth = node.count;
            lastCode = (lastCode + 1) << delta;
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

    _longestCode = 31 - countl_zero(lastCode);
}

void Compressor::transformRLE()
{
    _compressedData = reinterpret_cast<uint32_t *>(_dataPool);
    _compressedData[0] = 0;
    uint32_t currentCompressedIndex = 0;
    uint32_t nextCompressedIndex = 1;
    uint32_t imageIndex = 1;
    uint32_t sameSymbolCount = 1;
    symbol_t current = _fileData[0];
    uint8_t chunkIndex = 0;

    while (imageIndex <= _size)
    {
        if (_fileData[imageIndex] == current)
        {
            sameSymbolCount++;
        }
        else
        {
            cerr << "s: " << sameSymbolCount << " c: " << (char)current << endl;
            uint32_t code = _codeTable[current];
            bitset<32> codeBits(code);
            cerr << " c: " << codeBits << endl;
            uint32_t leadingZeros = countl_zero(code);
            uint32_t codeLength = 31 - leadingZeros;
            uint32_t mask = ~(1 << codeLength);
            uint32_t maskedCode = code & mask;
            bitset<32> maskedCodeBits(maskedCode);
            cerr << "mc: " << maskedCodeBits << endl;

            for (uint32_t i = 0; i < sameSymbolCount && i < 3; i++)
            {
                uint32_t upShiftedCode = maskedCode << chunkIndex;
                uint32_t downShiftedCode = maskedCode >> (32 - chunkIndex);
                _compressedData[currentCompressedIndex] |= upShiftedCode;
                _compressedData[nextCompressedIndex] = 0;
                _compressedData[nextCompressedIndex] |= downShiftedCode;
                chunkIndex += codeLength;
                bool moveChunk = chunkIndex >= 32;
                chunkIndex &= 31;
                currentCompressedIndex += moveChunk;
                nextCompressedIndex += moveChunk;
            }

            bool repeating = sameSymbolCount >= 3;
            sameSymbolCount -= 3;
            while (repeating)
            {
                uint32_t count = sameSymbolCount & 0x7F;
                sameSymbolCount >>= 7;
                repeating = sameSymbolCount > 0;
                count |= repeating << 7;
                uint32_t upShiftedCount = count << chunkIndex;
                uint32_t downShiftedCount = count >> (32 - chunkIndex);
                _compressedData[currentCompressedIndex] |= upShiftedCount;
                _compressedData[nextCompressedIndex] = 0;
                _compressedData[nextCompressedIndex] |= downShiftedCount;
                chunkIndex += 8;
                bool moveChunk = chunkIndex >= 32;
                chunkIndex &= 31;
                currentCompressedIndex += moveChunk;
                nextCompressedIndex += moveChunk;
            }

            sameSymbolCount = 1;
            current = _fileData[imageIndex];
        }
        imageIndex++;
    }
    _compressedSize = nextCompressedIndex;
}

void Compressor::createHeader()
{
    if (_longestCode >= 16)
    {
        StaticHeader32 header;
        header.width = _width;
        header.blockSize = _compressedSize;
        header.headerType = HEADER_OPTIONS.STATIC | HEADER_OPTIONS.DIRECT | HEADER_OPTIONS.ALL_SYMBOLS | HEADER_OPTIONS.CODE_LENGTHS_32;
        header.version = 0;
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            // TODO: fill in the header
        }
    }
    else
    {
        StaticHeader16 header;
        header.width = _width;
        header.blockSize = _compressedSize;
        header.headerType = HEADER_OPTIONS.STATIC | HEADER_OPTIONS.DIRECT | HEADER_OPTIONS.ALL_SYMBOLS | HEADER_OPTIONS.CODE_LENGTHS_16;
        header.version = 0;
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            // TODO: fill in the header
        }
    }
}

void Compressor::printTree(uint16_t nodeIdx, uint16_t indent = 0)
{
    if (_tree[nodeIdx].isLeaf())
    {
        Leaf leaf = reinterpret_cast<Leaf *>(_tree)[nodeIdx];
        cerr << string(indent, ' ') <<  nodeIdx << ": Leaf: " << leaf.count << " " << (char)(leaf.symbol) << endl;
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

    if (_size > UINT32_MAX) // TODO: possibly allow larger files
    {
        cerr << "Error: The input file is too large." << endl;
        exit(1);
    }
    else if (_size == 0) // TODO: possibly allow empty files
    {
        cerr << "Error: The input file is empty. Nothing to compress here." << endl;
        exit(1);
    }

    _height = _size / _width;
    if (_height * _width != _size) // verify that file is rectangular
    {
        cerr << "Error: The input file size is not a multiple of the specified width." << endl;
        exit(1);
    }

    _dataPool = static_cast<symbol_t *>(aligned_alloc(64, _width * _height * sizeof(symbol_t) + DATA_POOL_PADDING + 1));
    if (_dataPool == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the image." << endl;
        exit(1);
    }
    _fileData = _dataPool + DATA_POOL_PADDING;

    inputFile.read(reinterpret_cast<char *>(_fileData), _size);
    inputFile.close();
    _fileData[_size] = ~_fileData[_size - 1]; // add a dummy symbol to ensure correct RLE encoding
}

void Compressor::compressStatic()
{
    computeHistogram();

    // sort the histogram by frequency of symbols in ascending order
    sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, [](const Leaf &a, const Leaf &b) { return a.symbol < b.symbol; });
    sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
    
    buildHuffmanTree();

    populateCodeTable();
    
    printTree(_treeIndex);

    for (size_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        if (_codeTable[i])
        {
            cerr << (void *)i << ": ";
            bitset<32> codeBitset(_codeTable[i]);
            cerr << codeBitset << endl;
        }
    }

    transformRLE();

    for (uint32_t i = 0; i < _compressedSize; i++)
    {
        bitset<32> codeBitset(_compressedData[i]);
        cerr << codeBitset << " ";
    }
    cerr << endl;
}
