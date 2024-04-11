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
    uint16_t idx = _sortedNodesHead;
    Node tail = _sortedNodes[_sortedNodesTail];
    while (idx > _sortedNodesHead && _sortedNodes[idx].count > tail.count)
    {
        _sortedNodes[idx + 1] = _sortedNodes[idx];
        idx--;
    }
    _sortedNodes[idx + 1] = tail;
}

void Compressor::populateCodeTable()
{
    vint8 lastCode = _mm256_set1_epi32(0);
    uint16_t delta = 0;


    while (_tree[_treeIndex].isNode())
    {
        _treeIndex--;
    }

    _codeTable[_tree[_treeIndex].right] = lastCode;
    
    for (uint16_t i = _treeIndex; i >= 1; i--)
    {
        if (_tree[i].isLeaf())
        {
            // TODO
            delta = 0;
        }
        else
        {
            delta++;
        }
    }
}

void Compressor::printTree(uint16_t nodeIdx, uint16_t indent = 0)
{
    if (_tree[nodeIdx].isLeaf())
    {
        Leaf leaf = reinterpret_cast<Leaf *>(_tree)[nodeIdx];
        cerr << string(indent, ' ') << "Leaf: " << leaf.count << " " << static_cast<char>(leaf.pixelValue) << endl;
    }
    else
    {
        cerr << string(indent, ' ') << "Node: " << _tree[nodeIdx].count << " " << _tree[nodeIdx].left << " " << _tree[nodeIdx].right << endl;
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

    _image = static_cast<pixel_t *>(aligned_alloc(64, _width * _height * sizeof(pixel_t)));
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

    _sortedNodesHead = NUMBER_OF_SYMBOLS >> 1; // TODO: solve issue with underflow
    _sortedNodesTail = _sortedNodesHead + 1;
    _sortedNodes[_sortedNodesHead] = nodeTree[3];
    _sortedNodes[_sortedNodesHead].left = 3;

    bool connectTree = false;

    while (histogramIndex < NUMBER_OF_SYMBOLS - 1)
    {
        if (_histogram[histogramIndex + 1].count < _sortedNodes[_sortedNodesHead].count) // node from 2 symbols
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
            }
            while (_sortedNodesHead < _sortedNodesTail && _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count);
            
            _sortedNodes[_sortedNodesTail] = nodeTree[_treeIndex];
            _sortedNodes[_sortedNodesTail].left = _treeIndex;
            fixSortedNodes();
            _sortedNodesTail++;
        }
        else // newly created node has the smallest count
        {
            _sortedNodes[--_sortedNodesHead] = nodeTree[_treeIndex];
            _sortedNodes[_sortedNodesHead].left = _treeIndex;
        }
    }

    if (histogramIndex < NUMBER_OF_SYMBOLS)
    {
        leafTree[++_treeIndex] = _histogram[histogramIndex];
        nodeTree[_treeIndex + 1].count = leafTree[_treeIndex].count + _sortedNodes[_sortedNodesHead].count;
        nodeTree[_treeIndex + 1].left = _treeIndex;
        nodeTree[_treeIndex + 1].right = _sortedNodes[_sortedNodesHead].left;
        _treeIndex++;
    }

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
}

