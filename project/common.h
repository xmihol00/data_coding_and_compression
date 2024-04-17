#ifndef _COMMON_H_
#define _COMMON_H_

#include "omp.h"

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

#if 1
    #define DEBUG_PRINT(value) std::cerr << value << std::endl
#else
    #define DEBUG_PRINT(value)
#endif

using symbol_t = uint8_t;
using uint16v16_t = __m256i;
using uint32v8_t = __m256i;
using uint32v16_t = __m512i;
using uint64v4_t = __m256i;
using uint64v8_t = __m512i;

class HuffmanRLECompression
{
public:
    HuffmanRLECompression() = default;
    HuffmanRLECompression(bool model, bool adaptive) : _model{model}, _adaptive{adaptive} { };
    HuffmanRLECompression(bool model, bool adaptive, uint64_t width) : _model{model}, _adaptive{adaptive}, _width{width} { };
    ~HuffmanRLECompression() = default;

protected:
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};
    static constexpr uint16_t MAX_LONG_CODE_LENGTH{32};
    static constexpr uint16_t MAX_SHORT_CODE_LENGTH{16};
    static constexpr uint16_t BLOCK_SIZE{8};
    static constexpr uint16_t MAX_NUM_THREADS{8};
    static constexpr uint16_t CACHE_LINE_SIZE{128};

    bool _model;
    bool _adaptive;
    uint64_t _width;
    uint64_t _height;
    uint64_t _size;

    uint32_t _blockCount;
    uint32_t _blocksPerRow;
    uint32_t _blocksPerColumn;

    enum AdaptiveTraversals
    {
        HORIZONTAL = 0,
        VERTICAL = 1,
    };

    static constexpr struct
    {
        uint8_t SERIALIZATION   = 0;
        uint8_t TRANSFORMATION  = 1;
        uint8_t CODE_TABLE_TYPE = 2;
        uint8_t CODE_LENGTHS    = 3;

    } HEADER_OPTIONS
    {
        .SERIALIZATION   = 0,
        .TRANSFORMATION  = 1,
        .CODE_TABLE_TYPE = 2,
        .CODE_LENGTHS    = 3,
    };

    static constexpr struct 
    {
        uint8_t STATIC           = 0 << HEADER_OPTIONS.SERIALIZATION;
        uint8_t ADAPTIVE         = 1 << HEADER_OPTIONS.SERIALIZATION;

        uint8_t DIRECT           = 0 << HEADER_OPTIONS.TRANSFORMATION;
        uint8_t MODEL            = 1 << HEADER_OPTIONS.TRANSFORMATION;

        uint8_t ALL_SYMBOLS      = 0 << HEADER_OPTIONS.CODE_TABLE_TYPE;
        uint8_t SELECTED_SYMBOLS = 1 << HEADER_OPTIONS.CODE_TABLE_TYPE;

        uint8_t CODE_LENGTHS_16  = 0 << HEADER_OPTIONS.CODE_LENGTHS;
        uint8_t CODE_LENGTHS_32  = 1 << HEADER_OPTIONS.CODE_LENGTHS;
    } HEADERS
    {
        .STATIC           = 0 << HEADER_OPTIONS.SERIALIZATION,
        .ADAPTIVE         = 1 << HEADER_OPTIONS.SERIALIZATION,

        .DIRECT           = 0 << HEADER_OPTIONS.TRANSFORMATION,
        .MODEL            = 1 << HEADER_OPTIONS.TRANSFORMATION,

        .ALL_SYMBOLS      = 0 << HEADER_OPTIONS.CODE_TABLE_TYPE,
        .SELECTED_SYMBOLS = 1 << HEADER_OPTIONS.CODE_TABLE_TYPE,

        .CODE_LENGTHS_16  = 0 << HEADER_OPTIONS.CODE_LENGTHS,
        .CODE_LENGTHS_32  = 1 << HEADER_OPTIONS.CODE_LENGTHS,
    };

    struct BaseHeader
    {
        uint32_t width;
        uint32_t blockSize;
        uint8_t headerType;
        uint8_t version;
    } __attribute__((packed));

    static constexpr uint16_t CODE_LENGTHS_HEADER_SIZE{NUMBER_OF_SYMBOLS / 2};
    struct CodeLengthsHeader : public BaseHeader
    {
        uint8_t codeLengths[];
    } __attribute__((packed));

    struct DepthBitmapsHeader : public BaseHeader
    {
        uint16_t codeDepths;
    } __attribute__((packed));

    struct FullHeader : public BaseHeader
    {
        uint8_t buffer[256];
    } __attribute__((packed));

    static constexpr uint16_t ADDITIONAL_HEADER_SIZES[16] = {
        CODE_LENGTHS_HEADER_SIZE,
    };

    uint32_t _usedDepths;
    uint64v4_t _symbolsAtDepths[MAX_LONG_CODE_LENGTH] __attribute__((aligned(64)));

    inline constexpr void transposeBlock(symbol_t *source, uint32_t blockIdx)
    {
        uint32_t blockRow = blockIdx / _blocksPerRow;
        uint32_t blockColumn = blockIdx % _blocksPerRow;

        uint32_t startingIdx = blockRow * _width * BLOCK_SIZE + blockColumn * BLOCK_SIZE;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            uint32_t rowIdx = startingIdx + i * _width;
            uint32_t columnIdx = startingIdx + i;
            for (uint32_t j = 0; j < i; j++)
            {
                std::swap(source[rowIdx], source[columnIdx]);
                rowIdx++;
                columnIdx += _width;
            }
        }
    }
};

#endif // _COMMON_H_