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
    void parseBitmapHuffmanTree(uint16_t &readBytes);
    void parseThreadingInfo();
    void parseHeader();
    void transformRLE(uint16_t *compressedData, symbol_t *decompressedData, uint64_t bytesToDecompress);
    void reverseDifferenceModel(symbol_t *source, symbol_t *destination, uint64_t bytesToProcess);

    void decompressStatic();
    void decompressAdaptive();
    void decompressStaticModel();
    void decompressAdaptiveModel();

    uint8_t _numberOfCompressedBlocks{1};
    uint8_t _bitsPerCompressedBlockSize;
    uint32_t _numberOfBytesCompressedBlocks;
    uint8_t *_rawBlockSizes{reinterpret_cast<uint8_t *>(_compressedSizesExScan)}; // reuse already allocated memory

    uint8_t *_decompressionBuffer{nullptr};
    uint16_t *_compressedData{nullptr};
    symbol_t *_decompressedData{nullptr};
    uint8_t _rawDepthBitmaps[MAX_NUMBER_OF_CODES * NUMBER_OF_SYMBOLS];

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
    } _indexPrefixLengthCodeCount[MAX_NUMBER_OF_CODES] __attribute__((aligned(64)));

    struct IndexPrefixLength
    {
        uint16_t index;
        int16_t prefixLength;
    } _indexPrefixLengths[MAX_NUMBER_OF_CODES] __attribute__((aligned(64)));

    struct DepthIndices
    {
        uint8_t symbolsAtDepthIndex;
        uint8_t masksIndex;
        uint8_t depth;
        uint8_t prefixLength;
    } _depthsIndices[MAX_NUMBER_OF_CODES] __attribute__((aligned(64)));

    uint32_t _codeTableLarge[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));
    uint16_t *_codeTableSmall{reinterpret_cast<uint16_t *>(_codeTableLarge)};
    uint8_t _symbolsTable[NUMBER_OF_SYMBOLS * 2] __attribute__((aligned(64)));

    uint16_t _prefixIndices[MAX_NUMBER_OF_CODES];
    uint16_t _prefixShifts[MAX_NUMBER_OF_CODES];
    uint16_t _suffixShifts[MAX_NUMBER_OF_CODES];

    uint16v16_t _codePrefixesVector __attribute__((aligned(64)));
    uint16_t *_codePrefixes{reinterpret_cast<uint16_t *>(&_codePrefixesVector)};

    uint16v16_t _codeMasksVector;
    uint16_t *_codeMasks{reinterpret_cast<uint16_t *>(&_codeMasksVector)};
};

#endif // _DECOMPRESSOR_H_