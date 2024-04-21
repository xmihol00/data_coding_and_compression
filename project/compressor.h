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
    void applyDiferenceModel(symbol_t *source, symbol_t *destination);

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

    struct FrequencySymbolIndex
    {
        uint8_t index;
        uint8_t frequencyLowBits;
        uint16_t frequencyMidBits;
        uint32_t frequencyHighBits;
    } __attribute__((packed));

    struct SymbolParentDepth
    {
        uint8_t symbol;
        uint8_t depth;
        uint16_t parent;
    } __attribute__((packed));

    struct HuffmanCode
    {
        uint16_t code;
        uint16_t length;
    } __attribute__((packed));

    uint8_t _memoryPool[6 * NUMBER_OF_SYMBOLS * sizeof(uint64_t)] __attribute__((aligned(64)));
    FrequencySymbolIndex *_structHistogram{reinterpret_cast<FrequencySymbolIndex *>(_memoryPool)};
    uint64v8_t *_vectorHistogram{reinterpret_cast<uint64v8_t *>(_memoryPool)};
    uint64_t *_intHistogram{reinterpret_cast<uint64_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    SymbolParentDepth *_symbolsParentsDepths{reinterpret_cast<SymbolParentDepth *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    uint16_t *_parentsSortedIndices{reinterpret_cast<uint16_t *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    HuffmanCode *_codeTable{reinterpret_cast<HuffmanCode *>(_memoryPool)};
    symbol_t *_symbols{reinterpret_cast<symbol_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode))};
    uint8_t  *_depths{reinterpret_cast<uint8_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode) + NUMBER_OF_SYMBOLS * sizeof(symbol_t))};
    
    uint16_t _numberOfSymbols;
    uint16_t _treeIndex;
    uint16_t _sortedNodesHead;
    uint16_t _sortedNodesTail;

    uint8_t *_sourceBuffer{nullptr};
    uint8_t *_destinationBuffer{nullptr};

    uint8_t _longestCode;

    uint16_t _headerSize;
    uint16_t _threadBlocksSizesSize{0};

    int32_t *_rlePerBlockCounts[MAX_NUM_THREADS] = {nullptr, };

    uint64_t _repetitions[NUMBER_OF_SYMBOLS] = {0}; // TODO
};

#endif