#ifndef _DECOMPRESSOR_H_
#define _DECOMPRESSOR_H_

#include <string>

#include "common.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>

class Decompressor : public HuffmanRLECompression
{
public:
    Decompressor() = default;
    ~Decompressor();
    void decompress(std::string inputFileName, std::string outputFileName);

private:
    void readInputFile(std::string inputFileName);
    void parseHeader();

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

    uint32_t _largeCodeTable[NUMBER_OF_SYMBOLS] __attribute__((aligned(64)));
    uint16_t *_smallCodeTable{reinterpret_cast<uint16_t *>(_largeCodeTable)};

    FullHeader _header;
};

#endif