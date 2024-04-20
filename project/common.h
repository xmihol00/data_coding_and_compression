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

#define _DEBUG_PRINT_ACTIVE_
#ifdef _DEBUG_PRINT_ACTIVE_
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
    HuffmanRLECompression(bool model = false, bool adaptive = false, uint64_t width = 0, int32_t numberOfThreads = 1) 
        : _model{model}, _adaptive{adaptive}, _width{width}, _numberOfThreads{numberOfThreads} { };
    ~HuffmanRLECompression()
    {
        if (_blockTypes != nullptr)
        {
            delete[] _blockTypes;
        }

        if (_bestBlockTraversals != nullptr)
        {
            delete[] _bestBlockTraversals;
        }
    }

protected:
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};
    static constexpr uint16_t MAX_CODE_LENGTH{15};
    static constexpr uint16_t MAX_NUMBER_OF_CODES{16};
    static constexpr uint16_t BLOCK_SIZE{16};
    static constexpr uint16_t MAX_NUM_THREADS{32};
    static constexpr uint16_t BITS_PER_REPETITION_NUMBER{8};
    static constexpr uint16_t BITS_PER_BLOCK_TYPE{2};

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

    bool _model;
    bool _adaptive;
    uint64_t _width;
    uint64_t _height;
    uint64_t _size;

    uint32_t _blockCount;
    uint32_t _blocksPerRow;
    uint32_t _blocksPerColumn;

    int32_t _numberOfThreads;

    uint32_t _compressedSizes[MAX_NUM_THREADS];
    uint32_t _compressedSizesExScan[MAX_NUM_THREADS + 1];

    uint32_t _usedDepths;
    uint64v4_t _symbolsAtDepths[MAX_CODE_LENGTH] __attribute__((aligned(64)));

    uint8_t _headerBuffer[32];

    AdaptiveTraversals *_bestBlockTraversals{nullptr};

    uint8_t *_blockTypes{nullptr};
    uint32_t _blockTypesByteSize{0};
    
    struct FirstByteHeader
    {
    public:
        inline constexpr bool getCompressed() const { return _data & 0b1000'0000; }
        inline constexpr uint8_t getVersion() const { return (_data & 0b0110'0000) >> 5; }
        inline constexpr uint8_t getHeaderType() const { return _data & 0b0000'1111; }

        inline constexpr void clearFirstByte() { _data = 0; }
        inline constexpr void setNotCompressed() { _data |= 1 << 7; }
        inline constexpr void setVersion(uint8_t version) { _data = (_data & 0b1001'1111) | (version << 6) >> 1; }
        inline constexpr void insertHeaderType(uint8_t headerType) { _data |= headerType; }
    private:
        uint8_t _data;
    } __attribute__((packed));

    struct BasicHeader : public FirstByteHeader
    {
    public:
        inline constexpr void setSize(uint64_t size) 
        { 
            _baseWidth = size;
            _baseHeight = size >> 16;
            _extraWidth = size >> 32;
            _extraHeight = size >> 40;
        }
        inline constexpr uint64_t getSize() 
        { 
            return (static_cast<uint64_t>(_extraHeight) << 40) |
                   (static_cast<uint64_t>(_extraWidth) << 32) |
                   (static_cast<uint64_t>(_baseHeight) << 16) | 
                   _baseWidth; 
        }

        inline constexpr void setWidth(uint32_t width) 
        { 
            _baseWidth = width; 
            _extraWidth = width >> 16;
        }
        inline constexpr uint32_t getWidth() 
        { 
            return (static_cast<uint64_t>(_extraWidth) << 16) | _baseWidth; 
        }

        inline constexpr void setHeight(uint32_t height) 
        { 
            _baseHeight = height; 
            _extraHeight = height >> 16;
        }
        inline constexpr uint32_t getHeight() 
        { 
            return (static_cast<uint64_t>(_extraHeight) << 16) | _baseHeight; 
        }

    private:
        uint8_t _extraWidth;
        uint8_t _extraHeight;
        uint16_t _baseWidth;
        uint16_t _baseHeight;
    } __attribute__((packed));

    struct MultiThreadedHeader : public BasicHeader
    {
    public:
        inline constexpr uint8_t getNumberOfThreads() const { return _numberOfThreads; }
        inline constexpr void setNumberOfThreads(uint8_t numberOfThreads) { _numberOfThreads = numberOfThreads; }

        inline constexpr void setBitsPerBlockSize(uint8_t bitsPerBlockSize) { _bitsPerBlockSize = bitsPerBlockSize;}
        inline constexpr uint8_t getBitsPerBlockSize() { return _bitsPerBlockSize; }
    
    private:
        uint8_t _numberOfThreads;
        uint8_t _bitsPerBlockSize;
    } __attribute__((packed));

    struct DepthBitmapsHeader : public BasicHeader
    {
    public:
        inline constexpr uint16_t getCodeDepths() const { return _codeDepths; }
        inline constexpr void setCodeDepths(uint16_t codeDepths) { _codeDepths = codeDepths; }

    private:
        uint16_t _codeDepths;
    } __attribute__((packed));

    struct DepthBitmapsMultiThreadedHeader : public MultiThreadedHeader
    {
    public:
        inline constexpr uint16_t getCodeDepths() const { return _codeDepths; }
        inline constexpr void setCodeDepths(uint16_t codeDepths) { _codeDepths = codeDepths; }

    private:
        uint16_t _codeDepths;
    } __attribute__((packed));

    inline constexpr void initializeBlockTypes()
    {
        _blocksPerRow = (_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        _blocksPerColumn = (_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
        _blockCount = _blocksPerRow * _blocksPerColumn;
        _bestBlockTraversals = new AdaptiveTraversals[_blockCount + 4];
        reinterpret_cast<uint32_t *>(_bestBlockTraversals + _blockCount)[0] = 0; // clear the last 4 bytes (padding)
    }

    inline constexpr void transposeSerializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            for (uint32_t j = 0; j < BLOCK_SIZE; j++)
            {
                destination[destinationIdx++] = source[sourceStartingIdx + j * _width + i];
            }
        }
    }

    inline constexpr void transposeDeserializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            for (uint32_t j = 0; j < BLOCK_SIZE; j++)
            {
                destination[destinationIdx + j * _width + i] = source[sourceStartingIdx++];
            }
        }
    }

    inline constexpr void serializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        #pragma GCC unroll BLOCK_SIZE
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            // copy whole row at once
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

    inline constexpr void deserializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn)
    {
        uint32_t sourceStartingIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        uint32_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;
        #pragma GCC unroll BLOCK_SIZE
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {   
            // copy whole row at once
            if constexpr (BLOCK_SIZE == 8)
            {
                reinterpret_cast<uint8v8_t *>(destination + destinationIdx + i * _width)[0] = reinterpret_cast<uint8v8_t *>(source + sourceStartingIdx)[0];
            }
            else if constexpr (BLOCK_SIZE == 16)
            {
                _mm_store_si128(reinterpret_cast<uint8v16_t*>(destination + destinationIdx + i * _width), reinterpret_cast<uint8v16_t *>(source + sourceStartingIdx)[0]);
            }
            else if constexpr (BLOCK_SIZE == 32)
            {
                _mm256_store_si256(reinterpret_cast<uint8v32_t*>(destination + destinationIdx + i * _width), reinterpret_cast<uint8v32_t *>(source + sourceStartingIdx)[0]);
            }
            else if constexpr (BLOCK_SIZE == 64)
            {
                _mm512_store_si512(reinterpret_cast<uint8v64_t*>(destination + destinationIdx + i * _width), reinterpret_cast<uint8v64_t *>(source + sourceStartingIdx)[0]);
            }
            sourceStartingIdx += BLOCK_SIZE;
        }
    }
};

#endif // _COMMON_H_