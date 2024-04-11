#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

using pixel_t = uint8_t;
using vint8 = __m256i;

class Compressor
{
public:
    Compressor(bool model, bool adaptive, uint32_t width);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    static constexpr uint32_t NUMBER_OF_SYMBOLS{256};
    static constexpr vint8 ONE{reinterpret_cast<vint8>((__v8si){0, 0, 0, 0, 0, 0, 0, 1})};

    void readInputFile(std::string inputFileName);
    void computeHistogram();
    void compressStatic();
    void compressAdaptive();
    void compressStaticModel();
    void compressAdaptiveModel();
    void fixSortedNodes();
    void populateCodeTable();

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

    struct Canonical
    {
        uint8_t codeLength;
        uint16_t code;
    };

    uint8_t _memoryPool[NUMBER_OF_SYMBOLS * NUMBER_OF_SYMBOLS + 2 * NUMBER_OF_SYMBOLS] __attribute__((aligned(64))) = {0, };
    Node *_tree{reinterpret_cast<Node *>(_memoryPool)};
    Leaf *_histogram{reinterpret_cast<Leaf *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS)};
    Node *_sortedNodes{reinterpret_cast<Node *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS)};
    vint8 *_codeTable{reinterpret_cast<vint8 *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS)};
    
    uint16_t _treeIndex;
    uint16_t _sortedNodesHead;
    uint16_t _sortedNodesTail;

    pixel_t *_image{nullptr};
};

#endif