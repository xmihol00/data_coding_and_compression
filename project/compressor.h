#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "common.h"

class Compressor : public HuffmanRLECompression
{
public:
    Compressor(bool model, bool adaptive, uint64_t width);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    static constexpr uint16_t COMPRESSED_ORIGINAL_OFFSET{256};
    
    void readInputFile(std::string inputFileName);
    void computeHistogram();
    void buildHuffmanTree();
    void populateCodeTable();
    void transformRLE();
    void createHeader();
    void writeOutputFile(std::string outputFileName);

    void compressStatic();
    void compressAdaptive();
    void compressStaticModel();
    void compressAdaptiveModel();

    void analyzeImageAdaptive();

    void printTree(uint16_t nodeIdx, uint16_t indent);

    struct Leaf
    {
        uint32_t count;
        uint16_t : 16;
        uint8_t symbol;
        uint8_t : 8;
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

    uint8_t _memoryPool[6 * NUMBER_OF_SYMBOLS * sizeof(Node)] __attribute__((aligned(64)));
    Node *_tree{reinterpret_cast<Node *>(_memoryPool)};
    Leaf *_histogram{reinterpret_cast<Leaf *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    Node *_sortedNodes{reinterpret_cast<Node *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    uint32_t *_codeTable{reinterpret_cast<uint32_t *>(_memoryPool + 2 * NUMBER_OF_SYMBOLS * sizeof(Node))};
    
    uint16_t _treeIndex;
    uint16_t _sortedNodesHead;
    uint16_t _sortedNodesTail;

    uint8_t *_dataPool{nullptr};
    symbol_t *_fileData{nullptr};       // alias to _dataPool
    uint16_t *_compressedData{nullptr}; // alias to _dataPool
    symbol_t *_serializedData{nullptr}; // separate buffer for adaptive compression

    uint8_t _longestCode;
    uint32_t _compressedSize;

    FullHeader _header;
    uint16_t _headerSize;

    AdaptiveTraversals *_bestBlockTraversals{nullptr};
    int32_t *_rlePerBlockCounts[MAX_NUM_THREADS] = {nullptr, };
};

#endif