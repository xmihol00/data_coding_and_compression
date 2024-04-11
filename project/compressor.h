#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>

using pixel_t = uint8_t;

class Compressor
{
public:
    Compressor(bool model, bool adaptive, uint32_t width);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    static constexpr uint32_t NUMBER_OF_SYMBOLS = 256;

    void readInputFile(std::string inputFileName);
    void computeHistogram();
    void compressStatic();
    void compressAdaptive();
    void compressStaticModel();
    void compressAdaptiveModel();
    void fixSortedNodes();
    void printTree(uint16_t nodeIdx, uint16_t indent);

    bool _model;
    bool _adaptive;
    uint32_t _width;
    uint32_t _height;
    uint64_t _size;

    struct Leaf
    {
        uint32_t count;
        uint16_t : 16;
        uint8_t  : 8;
        pixel_t pixelValue;
    };

    struct Node
    {
        uint32_t count;
        uint16_t left;
        uint16_t right;

        constexpr inline bool isNode() const { return left; }
        constexpr inline bool isLeaf() const { return !left; }
    };

    Node test;

    Leaf _histogram[NUMBER_OF_SYMBOLS] = {0, };
    Node _tree[2 * NUMBER_OF_SYMBOLS] = {0, };

    uint16_t _sortedNodesHead = 0;
    uint16_t _sortedNodesTail = 0;
    Node _sortedNodes[2 * NUMBER_OF_SYMBOLS] = {0, };

    pixel_t *_image = nullptr;
};

#endif