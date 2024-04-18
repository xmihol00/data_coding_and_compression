#ifndef _DECOMPRESSOR_H_
#define _DECOMPRESSOR_H_

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <bitset>

#include "common.h"

class Decompressor : public HuffmanRLECompression
{
public:
    Decompressor(int32_t numberOfThreads);
    ~Decompressor();
    void decompress(std::string inputFileName, std::string outputFileName);

private:
    bool readInputFile(std::string inputFileName, std::string outputFileName);
    void writeOutputFile(std::string outputFileName);
    void parseBitmapHuffmanTree();
    void parseThreadingInfo();
    void parseHeader();
    void decomposeDataBetweenThreads(uint64_t &bytesPerThread, uint32_t &startingIdx);
    void transformRLE(uint16_t *compressedData, symbol_t *decompressedData, uint64_t bytesToDecompress);

    void decompressStatic();

    uint32_t _width;
    uint32_t _size;

    uint8_t _numberOfCompressedBlocks;
    uint8_t _bitsPerCompressedBlockSize;
    uint32_t _numberOfBytesCompressedBlocks;
    uint8_t *_rawBlockSizes{reinterpret_cast<uint8_t *>(_compressedSizesExScan)}; // reuse already allocated memory

    uint64_t _decompressedSize;
    uint8_t *_decompressedData{nullptr};
    uint16_t *_compressedData{nullptr};

    FirstByteHeader &_header = reinterpret_cast<FirstByteHeader &>(_headerBuffer);

    struct CodeLengthSymbol
    {
        uint8_t codeLength;
        uint8_t symbol;
    } _codeLengthsSymbols[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));

    struct IndexPrefixLengthCodeCount
    {
        uint16_t codeCount;
        uint16_t cumulativeCodeCount;
        uint16_t index;
        int16_t prefixLength;
    } _indexPrefixLengthCodeCount[MAX_SHORT_CODE_LENGTH] __attribute__((aligned(64)));

    struct IndexPrefixLength
    {
        uint16_t index;
        int16_t prefixLength;
    } _indexPrefixLengths[MAX_SHORT_CODE_LENGTH] __attribute__((aligned(64)));

    struct DepthIndices
    {
        uint8_t symbolsAtDepthIndex;
        uint8_t masksIndex;
        uint8_t depth;
        uint8_t prefixLength;
    } _depthsIndices[MAX_SHORT_CODE_LENGTH] __attribute__((aligned(64)));

    uint32_t _codeTableLarge[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));
    uint16_t *_codeTableSmall{reinterpret_cast<uint16_t *>(_codeTableLarge)};
    uint8_t _symbolsTable[NUMBER_OF_SYMBOLS * 2] __attribute__((aligned(64)));

    uint16_t _prefixIndices[MAX_LONG_CODE_LENGTH];
    uint16_t _prefixShifts[1 << (sizeof(uint8_t) * 8)];
    uint16_t _suffixShifts[1 << (sizeof(uint8_t) * 8)];

    uint32v16_t _codePrefixesLargeVector[MAX_LONG_CODE_LENGTH / 16] __attribute__((aligned(64)));
    uint16v16_t *_codePrefixesSmallVector{reinterpret_cast<uint16v16_t *>(_codePrefixesLargeVector)};
    uint32_t *_codePrefixesLarge{reinterpret_cast<uint32_t *>(_codePrefixesLargeVector)};
    uint16_t *_codePrefixesSmall{reinterpret_cast<uint16_t *>(_codePrefixesLargeVector)};

    uint32v16_t _codeMasksLargeVector[MAX_LONG_CODE_LENGTH / 4] __attribute__((aligned(64)));
    uint16v16_t *_codeMasksSmallVector{reinterpret_cast<uint16v16_t *>(_codeMasksLargeVector)};
    uint32_t *_codeMasksLarge{reinterpret_cast<uint32_t *>(_codeMasksLargeVector)};
    uint16_t *_codeMasksSmall{reinterpret_cast<uint16_t *>(_codeMasksLargeVector)};
};

#endif