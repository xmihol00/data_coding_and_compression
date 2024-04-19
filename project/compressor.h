#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "common.h"

class Compressor : public HuffmanRLECompression
{
public:
    Compressor(bool model, bool adaptive, uint64_t width, int32_t numberOfThreads);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    void readInputFile(std::string inputFileName);
    void computeHistogram();
    void buildHuffmanTree();
    void populateCodeTable();
    void decomposeDataBetweenThreads(symbol_t *data, uint32_t &bytesPerThread, uint32_t &startingIdx, symbol_t &firstSymbol);
    void transformRLE(symbol_t *sourceData, uint16_t *compressedData, uint32_t &compressedSize, uint64_t &startingIdx);
    void createHeader();
    void writeOutputFile(std::string outputFileName, std::string inputFileName);

    void compressStatic();
    void compressAdaptive();
    void compressStaticModel();
    void compressAdaptiveModel();

    void analyzeImageAdaptive();
    void applyDiferenceModel(symbol_t firstSymbol, symbol_t *source, symbol_t *destination, uint32_t bytesToProcess);

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

    symbol_t *_fileData{nullptr};
    uint16_t *_compressedData{nullptr};
    symbol_t *_serializedData{nullptr};

    uint8_t _longestCode;

    uint16_t _headerSize;
    uint16_t _threadBlocksSizesSize{0};

    int32_t *_rlePerBlockCounts[MAX_NUM_THREADS] = {nullptr, };
};

#endif