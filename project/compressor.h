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
    void clearMemory();
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

    static constexpr FrequencySymbolIndex MAX_FREQUENCY_SYMBOL_INDEX = { .index = 0xff, .frequencyLowBits = 0xff, .frequencyMidBits = 0xffff, .frequencyHighBits = 0x7fff'ffff };

    uint8_t _memoryPool[10 * NUMBER_OF_SYMBOLS * sizeof(uint64_t)] __attribute__((aligned(64)));
    FrequencySymbolIndex *_structHistogram{reinterpret_cast<FrequencySymbolIndex *>(_memoryPool)};
    uint64v8_t *_vectorHistogram{reinterpret_cast<uint64v8_t *>(_memoryPool)};
    uint64_t *_intHistogram{reinterpret_cast<uint64_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    SymbolParentDepth *_symbolsParentsDepths{reinterpret_cast<SymbolParentDepth *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    uint16_t *_parentsSortedIndices{reinterpret_cast<uint16_t *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(FrequencySymbolIndex))};
    HuffmanCode *_codeTable{reinterpret_cast<HuffmanCode *>(_memoryPool)};
    symbol_t *_symbols{reinterpret_cast<symbol_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode))};
    uint8_t  *_depths{reinterpret_cast<uint8_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode) + NUMBER_OF_SYMBOLS * sizeof(symbol_t))};
    
    bool _compressionUnsuccessful{false};
    uint16_t _numberOfSymbols;

    uint8_t *_sourceBuffer{nullptr};
    uint8_t *_destinationBuffer{nullptr};

    uint8_t _longestCode;

    uint16_t _headerSize;
    uint16_t _threadBlocksSizesSize{0};

    int32_t *_rlePerBlockCounts[MAX_NUM_THREADS] = {nullptr, };

    uint64_t _repetitions[NUMBER_OF_SYMBOLS];

    uint64_t _threadPadding;
};

#endif