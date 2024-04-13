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
#include <queue>
#include <bitset>
#include <bit>

#include "common.h"

using symbol_t = uint8_t;

class Compressor
{
public:
    Compressor(bool model, bool adaptive, uint64_t width);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    static constexpr uint16_t DATA_POOL_PADDING{64};
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};

    void readInputFile(std::string inputFileName);
    void computeHistogram();
    void buildHuffmanTree();
    void populateCodeTable();
    void transformRLE();
    void createHeader();

    void compressStatic();
    void compressAdaptive();
    void compressStaticModel();
    void compressAdaptiveModel();

    void printTree(uint16_t nodeIdx, uint16_t indent);

    bool _model;
    bool _adaptive;
    uint64_t _width;
    uint64_t _height;
    uint64_t _size;

    struct Leaf
    {
        uint32_t count;
        uint16_t : 16;
        symbol_t symbol;
        uint8_t  : 8;
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

    uint8_t _memoryPool[6 * NUMBER_OF_SYMBOLS * sizeof(Node)] __attribute__((aligned(64))) = {0, };
    Node *_tree{reinterpret_cast<Node *>(_memoryPool)};
    Leaf *_histogram{reinterpret_cast<Leaf *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    Node *_sortedNodes{reinterpret_cast<Node *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    uint32_t *_codeTable{reinterpret_cast<uint32_t *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    
    uint16_t _treeIndex;
    uint16_t _sortedNodesHead;
    uint16_t _sortedNodesTail;

    uint8_t *_dataPool{nullptr};
    symbol_t *_fileData{nullptr};
    uint32_t *_compressedData{nullptr};

    uint8_t _longestCode;
    uint32_t _compressedSize;
};

#endif