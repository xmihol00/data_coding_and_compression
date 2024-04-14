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
    Decompressor() = default;
    ~Decompressor();
    void decompress(std::string inputFileName, std::string outputFileName);

private:
    void readInputFile(std::string inputFileName);
    void parseHeader();
    void writeOutputFile(std::string outputFileName);

    void decompressStatic();

    uint32_t _width;
    uint32_t _size;
    uint8_t _headerType;

    uint8_t *_decompressedData{nullptr};
    uint32_t *_compressedData{nullptr};

    struct CodeLengthSymbol
    {
        uint8_t codeLength;
        uint8_t symbol;
    } _codeLengthsSymbols[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));

    uint32_t _codeTableLarge[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));
    uint16_t *_codeTableSmall{reinterpret_cast<uint16_t *>(_codeTableLarge)};
    uint8_t _symbolsTable[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));

    FullHeader _header;

    uint8_t _prefixIndices[MAX_LONG_CODE_LENGTH];
    uint8_t _prefixLengths[1 << (sizeof(uint8_t) * 8)];
    uint8_t _suffixShifts[1 << (sizeof(uint8_t) * 8)];

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