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
using uint8v8_t = uint64_t;
using uint8v16_t = __m128i;
using uint8v32_t = __m256i;
using uint8v64_t = __m512i;
using uint32v8_t = __m256i;
using uint32v16_t = __m512i;
using uint64v4_t = __m256i;
using uint64v8_t = __m512i;

class HuffmanRLECompression
{
public:
    HuffmanRLECompression(bool model = false, bool adaptive = false, uint64_t width = 0, int32_t numThreads = 1) 
        : _model{model}, _adaptive{adaptive}, _width{width}, _numThreads{numThreads} { };
    ~HuffmanRLECompression() = default;

protected:
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};
    static constexpr uint16_t MAX_LONG_CODE_LENGTH{32};
    static constexpr uint16_t MAX_SHORT_CODE_LENGTH{16};
    static constexpr uint16_t BLOCK_SIZE{16};
    static constexpr uint16_t MAX_NUM_THREADS{32};
    static constexpr uint16_t CACHE_LINE_SIZE{128};

    bool _model;
    bool _adaptive;
    uint64_t _width;
    uint64_t _height;
    uint64_t _size;

    uint32_t _blockCount;
    uint32_t _blocksPerRow;
    uint32_t _blocksPerColumn;

    int32_t _numThreads;
    
    enum AdaptiveTraversals
    {
        HORIZONTAL = 0,
        VERTICAL = 1,
    };

    enum HeaderOptions
    {
        SERIALIZATION = 0,
        TRANSFORMATION,
        THREADS,
    };

    enum HeaderValues
    {
        STATIC          = 0 << SERIALIZATION,
        ADAPTIVE        = 1 << SERIALIZATION,
        DIRECT          = 0 << TRANSFORMATION,
        MODEL           = 1 << TRANSFORMATION,
        SINGLE_THREADED = 0 << THREADS,
        MULTI_THREADED  = 1 << THREADS
    };

    struct FirstByteHeader
    {
    public:
        inline constexpr bool getCompressed() const { return data & 0b1000'0000; }
        inline constexpr uint8_t getVersion() const { return (data & 0b0110'0000) >> 5; }
        inline constexpr uint8_t getHeaderType() const { return data & 0b0000'1111; }

        inline constexpr void clear() { data = 0; }
        inline constexpr void setNotCompressed() { data |= 1 << 7; }
        inline constexpr void setVersion(uint8_t version) { data = (data & 0b1001'1111) | (version << 6) >> 1; }
        inline constexpr void insertHeaderType(uint8_t headerType) { data |= headerType; }
    private:
        uint8_t data;
    } __attribute__((packed));

    struct SingleThreadedHeader : public FirstByteHeader
    {
    public:
        inline constexpr void setSize(uint64_t size) { extraSize = size >> 32; baseSize = size; }
        inline constexpr uint64_t getSize() { return (static_cast<uint64_t>(extraSize) << 32) | baseSize; }
    private:
        uint8_t extraSize;
        uint32_t baseSize;
    } __attribute__((packed));

    struct MultiThreadedHeader : public FirstByteHeader
    {
    public:
        inline constexpr uint8_t getNumThreads() const { return data & 0b0001'1111; }
        inline constexpr void setNumThreads(uint8_t numThreads) { data = (data & 0b1110'0000) | numThreads; }

        inline constexpr void setSize(uint64_t size) { extraSize = size >> 32; baseSize = size; }
        inline constexpr uint64_t getSize() { return (static_cast<uint64_t>(extraSize & 0b0001'1111) << 32) | baseSize; }

        inline constexpr void setBitsPerBlock(uint8_t bitsPerBlock) 
        { 
            extraSize = (extraSize & 0b0001'1111) | (bitsPerBlock << 5);
            data = (data & 0b0001'1111) | ((bitsPerBlock & 0b0011'100) << 2);
        }
        inline constexpr uint8_t getBitsPerBlock() { return ((extraSize & 0b1110'0000) >> 5) | ((data & 0b1110'0000) >> 2); }
    
    private:
        uint8_t extraSize;
        uint32_t baseSize;
        uint8_t data;
    } __attribute__((packed));

    static constexpr uint16_t CODE_LENGTHS_HEADER_SIZE{NUMBER_OF_SYMBOLS / 2};
    struct CodeLengthsSingleThreadedHeader : public SingleThreadedHeader
    {
        uint8_t codeLengths[];
    } __attribute__((packed));

    struct DepthBitmapsSingleThreadedHeader : public SingleThreadedHeader
    {
        uint16_t codeDepths;
    } __attribute__((packed));

    struct DepthBitmapsMultiThreadedHeader : public MultiThreadedHeader
    {
        uint16_t codeDepths;
    } __attribute__((packed));

    struct FullHeader : public SingleThreadedHeader
    {
        uint8_t buffer[256];
    } __attribute__((packed));

    static constexpr uint16_t ADDITIONAL_HEADER_SIZES[16] = {
        CODE_LENGTHS_HEADER_SIZE,
    };

    uint32_t _usedDepths;
    uint64v4_t _symbolsAtDepths[MAX_LONG_CODE_LENGTH] __attribute__((aligned(64)));

    inline constexpr void transposeSerializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * _width * BLOCK_SIZE + blockColumn * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            for (uint32_t j = 0; j < BLOCK_SIZE; j++)
            {
                destination[destinationIdx++] = source[sourceStartingIdx + j * _width + i];
            }
        }
    }

    inline constexpr void serializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * _width * BLOCK_SIZE + blockColumn * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        #pragma GCC unroll BLOCK_SIZE
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            // copy whole line at once
            if constexpr (BLOCK_SIZE == 8)
            {
                reinterpret_cast<uint8v8_t *>(destination + destinationIdx)[0] = reinterpret_cast<uint8v8_t *>(source + sourceStartingIdx + i * _width)[0];
            }
            else if constexpr (BLOCK_SIZE == 16)
            {
                reinterpret_cast<uint8v16_t *>(destination + destinationIdx)[0] = _mm_load_si128(reinterpret_cast<uint8v16_t*>(source + sourceStartingIdx + i * _width));
            }
            else if constexpr (BLOCK_SIZE == 32)
            {
                reinterpret_cast<uint8v32_t *>(destination + destinationIdx)[0] = _mm256_load_si256(reinterpret_cast<uint8v32_t*>(source + sourceStartingIdx + i * _width));
            }
            else if constexpr (BLOCK_SIZE == 64)
            {
                reinterpret_cast<uint8v64_t *>(destination + destinationIdx)[0] = _mm512_load_si512(reinterpret_cast<uint8v64_t*>(source + sourceStartingIdx + i * _width));
            }
            destinationIdx += BLOCK_SIZE;
        }
    }
};

#endif // _COMMON_H_