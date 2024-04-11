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
        _histogram[i].count = 0;
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
    memset(_tree, 0, sizeof(_tree));

    // sort the histogram by frequency in ascending order
    sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
    
    // create the Huffman tree starting from first character that occurred at least once
    uint16_t histogramIndex = 0;
    while (_histogram[histogramIndex].count == 0) // filter out characters that did not occur
    {
        histogramIndex++;
    }

    for (uint16_t i = histogramIndex; i < NUMBER_OF_SYMBOLS; i++)
    {
        cerr << _histogram[i].count << " " << static_cast<char>(_histogram[i].pixelValue) << endl;
    }

    uint16_t treeIndex = 3;
    Leaf *leafTree = reinterpret_cast<Leaf *>(_tree);
    Node *nodeTree = _tree;

    leafTree[1] = _histogram[histogramIndex++]; // TODO: solve only one character in histogram
    leafTree[2] = _histogram[histogramIndex++];
    nodeTree[3].count = leafTree[1].count + leafTree[2].count;
    nodeTree[3].left = 1;
    nodeTree[3].right = 2;

    _sortedNodesHead = NUMBER_OF_SYMBOLS >> 1; // TODO: solve issue with underflow
    _sortedNodesTail = _sortedNodesHead + 1;
    _sortedNodes[_sortedNodesHead] = nodeTree[3];
    _sortedNodes[_sortedNodesHead].left = 3;

    bool connectTree = false;

    while (histogramIndex < NUMBER_OF_SYMBOLS - 1)
    {
        if (_histogram[histogramIndex + 1].count < _sortedNodes[_sortedNodesHead].count)
        {
            cerr << "Double insert" << endl;
            leafTree[++treeIndex] = _histogram[histogramIndex++];
            leafTree[++treeIndex] = _histogram[histogramIndex++];
            
            ++treeIndex;
            nodeTree[treeIndex].count = leafTree[treeIndex - 2].count + leafTree[treeIndex - 1].count;
            nodeTree[treeIndex].left = treeIndex - 2;
            nodeTree[treeIndex].right = treeIndex - 1;

            connectTree = _sortedNodes[_sortedNodesHead].count <= nodeTree[treeIndex].count;
        }
        else
        {
            cerr << "Single insert " << _histogram[histogramIndex].count << " " << _sortedNodes[_sortedNodesHead].count << endl;
            leafTree[++treeIndex] = _histogram[histogramIndex++];
            connectTree = true;
        }

        if (connectTree)
        {
            cerr << "Tree connect" << endl;
            do
            {
                nodeTree[treeIndex + 1].count = _sortedNodes[_sortedNodesHead].count + nodeTree[treeIndex].count;
                nodeTree[treeIndex + 1].left = _sortedNodes[_sortedNodesHead++].left;
                nodeTree[treeIndex + 1].right = treeIndex;
                treeIndex++;
            }
            while (_sortedNodesHead < _sortedNodesTail && _sortedNodes[_sortedNodesHead].count <= nodeTree[treeIndex].count);
            for (uint16_t i = 1; i <= treeIndex; i++)
            {
                if (_tree[i].isLeaf())
                {
                    cerr << "Leaf: " << leafTree[i].count << " " << static_cast<char>(leafTree[i].pixelValue) << endl;
                }
                else
                {
                    cerr << "Node: " << nodeTree[i].count << " " << nodeTree[i].left << " " << nodeTree[i].right << endl;
                }
            }
            cerr << endl;

            _sortedNodes[_sortedNodesTail] = nodeTree[treeIndex];
            _sortedNodes[_sortedNodesTail].left = treeIndex;
            fixSortedNodes();
            _sortedNodesTail++;
            cerr << "Head: " << _sortedNodes[_sortedNodesHead].count << " " << _sortedNodesHead << " Tail: " << _sortedNodesTail << endl;
        }
        else
        {
            _sortedNodes[--_sortedNodesHead] = nodeTree[treeIndex];
            _sortedNodes[_sortedNodesHead].left = treeIndex;
        }
    }

    if (histogramIndex < NUMBER_OF_SYMBOLS)
    {
        leafTree[++treeIndex] = _histogram[histogramIndex];
        nodeTree[treeIndex + 1].count = leafTree[treeIndex].count + _sortedNodes[_sortedNodesHead].count;
        nodeTree[treeIndex + 1].left = treeIndex;
        nodeTree[treeIndex + 1].right = _sortedNodes[_sortedNodesHead].left;
        treeIndex++;
    }

    for (uint16_t i = 1; i <= treeIndex; i++)
    {
        if (_tree[i].isLeaf())
        {
            cerr << i << ": Leaf: " << leafTree[i].count << " " << static_cast<char>(leafTree[i].pixelValue) << endl;
        }
        else
        {
            cerr << i << ": Node: " << nodeTree[i].count << " " << nodeTree[i].left << " " << nodeTree[i].right << endl;
        }
    }
    cerr << endl;

    printTree(treeIndex);
}
