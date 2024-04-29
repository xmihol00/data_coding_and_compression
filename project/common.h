/* =======================================================================================================================================================
 * Project:         Huffman RLE compression and decompression
 * Author:          David Mihola (xmihol00)
 * E-mail:          xmihol00@stud.fit.vutbr.cz
 * Date:            12. 5. 2024
 * Description:     A parallel implementation of a compression algorithm based on Huffman coding and RLE transformation. 
 *                  This file contains base class with definitions inherited by both the compressor and decompressor.
 * ======================================================================================================================================================= */

#ifndef _COMMON_H_
#define _COMMON_H_

#if _OPENMP
    #include "omp.h"
#else
    /* dummy functions when OpenMP is not available */
    constexpr int omp_get_thread_num() { return 0; }
    constexpr int omp_get_num_threads() { return 1; }
    constexpr void omp_set_num_threads(int) { }
#endif

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

#ifdef _DEBUG_PRINT_ACTIVE_
    #define DEBUG_PRINT(value) std::cerr << value << std::endl
#else
    #define DEBUG_PRINT(value)
#endif

// used data types
using symbol_t = uint8_t;
using uint16v16_t = __m256i;
using uint16v32_t = __m512i;
using uint8v8_t = uint64_t;
using uint8v16_t = __m128i;
using uint8v32_t = __m256i;
using uint8v64_t = __m512i;
using uint32v8_t = __m256i;
using uint32v16_t = __m512i;
using uint64v4_t = __m256i;
using uint64v8_t = __m512i;

/**
 * @brief Base class for Huffman RLE compression from which the compressor and decompressor inherit.
 */
class HuffmanRLECompression
{
public:
    /**
     * @param model Flag indicating if the compression is model-based, unused for decompression.
     * @param adaptive Flag indicating if the compression is adaptive, unused for decompression.
     * @param width Width of the image to be compressed, unused for decompression.
     * @param numberOfThreads Number of threads used for multi-threaded compression/decompression.
     */
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

    /**
     * @brief Types of image traversal for adaptive compression.
     */
    enum AdaptiveTraversals
    {
        HORIZONTAL = 0,
        VERTICAL = 1,
    };

    /**
     * @brief Bit positions of specific header options in the compressed file.
     */
    enum HeaderOptions
    {
        SERIALIZATION = 0,
        TRANSFORMATION,
        THREADS,
    };

    /**
     * @brief Values of specific header options in the compressed file.
     */
    enum HeaderValues
    {
        STATIC          = 0 << SERIALIZATION,
        ADAPTIVE        = 1 << SERIALIZATION,
        DIRECT          = 0 << TRANSFORMATION,
        MODEL           = 1 << TRANSFORMATION,
        SINGLE_THREADED = 0 << THREADS,
        MULTI_THREADED  = 1 << THREADS
    };

    /**
     * @brief Maximum number of symbols that can be compressed (values from 0 to 255, i.e. 1 byte of information).
     */
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};

    /**
     * @brief Maximum length of a huffman code.
     */
    static constexpr uint16_t MAX_CODE_LENGTH{15};

    /**
     * @brief Maximum number of huffman code lengths.
     */
    static constexpr uint16_t MAX_NUMBER_OF_CODES{16};

    /**
     * @brief Maximum number of prefixes for codes at different depths of the huffman tree.
     */
    static constexpr uint16_t MAX_NUMBER_OF_PREFIXES{32};

    /**
     * @brief Block size for traversals during adaptive compression.
     */
    static constexpr uint16_t BLOCK_SIZE{16};

    /**
     * @brief Maximum number of threads that can be used for multi-threaded compression.
     */
    static constexpr uint16_t MAX_NUM_THREADS{32};

    /**
     * @brief Number of bits (size of a repetition chunk) used to encode repetitions of a character, the MSB is used to determine if the next chunk of bits is 
     *        the continuation of the number of repetitions (MSB is set), or the number of repetitions ends by this chunk (MSB is not set).
     */
    static constexpr uint16_t BITS_PER_REPETITION_NUMBER{4};

    /**
     * @brief Number of bits used to encode the type of a block (horizontal or vertical traversal) during adaptive compression.
     */
    static constexpr uint16_t BITS_PER_BLOCK_TYPE{2};

    bool _model;       ///< Flag indicating if the compression is model-based.
    bool _adaptive;    ///< Flag indicating if the compression is adaptive.
    uint64_t _width;   ///< Width of the image to be compressed/decompressed.
    uint64_t _height;  ///< Height of the image to be compressed/decompressed.
    uint64_t _size;    ///< Overall size of the image to be compressed/decompressed.

    uint32_t _blockCount;       ///< Total number of blocks in an image.
    uint32_t _blocksPerRow;     ///< Number of blocks in a row of an image.
    uint32_t _blocksPerColumn;  ///< Number of blocks in a column of an image.

    int32_t _numberOfThreads;   ///< Number of active threads used for multi-threaded compression.

    uint32_t _compressedSizes[MAX_NUM_THREADS + 1];         ///< Sizes of compressed data by each thread.
    uint32_t _compressedSizesExScan[MAX_NUM_THREADS + 1];   ///< Exclusive scan of compressed sizes by each thread.

    uint8_t _symbolsPerDepth[MAX_NUMBER_OF_CODES * 2];      ///< Number of symbols at each depth of the huffman tree.
    uint32_t _usedDepths;                                   ///< Bitmap of used depths in the final huffman tree (rebalanced if needed).
    uint64v4_t _symbolsAtDepths[MAX_NUMBER_OF_CODES * 2] __attribute__((aligned(64))); ///< NUmber of symbols at each depth of the final huffman tree.

    uint8_t _headerBuffer[32];  ///< Memory buffer to store a specific header depending on the type of the compression and number of threads.

    AdaptiveTraversals *_bestBlockTraversals{nullptr}; ///< Dynamically allocated array with the best traversal option for each block during adaptive compression.

    uint8_t *_blockTypes{nullptr};   ///< Dynamically allocated array with packed best traversal options for each block during adaptive compression.
    uint32_t _blockTypesByteSize{0}; ///< Size of the packed block types array in bytes.
    
    /**
     * @brief Structure representing the first byte of the compressed file header.
     *        Based on this byte, the correct header type can be determined and loaded.
     */
    struct FirstByteHeader
    {
    public:
        /**
         * @brief Checks whether the compressed data is compressed or not. When false, the data is not compressed and all the following bytes are raw data
         *        of the original file.
         */
        inline constexpr bool getCompressed() const { return _data & 0b1000'0000; }

        /**
         * @brief Gets the version of the used compression. The version is currently not used and is always set to 0.
         */
        inline constexpr uint8_t getVersion() const { return (_data & 0b0110'0000) >> 5; }

        /**
         * @brief Gets the type of the header that follows the first byte.
         */
        inline constexpr uint8_t getHeaderType() const { return _data & 0b0000'1111; }

        /**
         * @brief Clears the first byte of the header.
         */
        inline constexpr void clearFirstByte() { _data = 0; }

        /**
         * @brief Sets the not compressed flag to true, i.e. the data is not compressed.
         */
        inline constexpr void setNotCompressed() { _data |= 1 << 7; }

        /**
         * @brief Sets the version of the used compression.
         */
        inline constexpr void setVersion(uint8_t version) { _data = (_data & 0b1001'1111) | (version << 6) >> 1; }

        /**
         * @brief Inserts the type of the header that follows the first byte. Note, this function does not clear previously set header types.
         * @param headerType Only the lower 4 bits should be set.
         */
        inline constexpr void insertHeaderType(uint8_t headerType) { _data |= headerType; }
    private:
        uint8_t _data;
    } __attribute__((packed));

    /**
     * @brief Base header type, which is inherited to further extend the header of the compressed file.
     */
    struct BasicHeader : public FirstByteHeader
    {
    public:
        /**
         * @brief Sets the size of the original uncompressed data.
         */
        inline constexpr void setSize(uint64_t size) 
        { 
            _baseWidth = size;
            _baseHeight = size >> 16;
            _extraWidth = size >> 32;
            _extraHeight = size >> 40;
        }

        /**
         * @brief Gets the size of the original uncompressed data.
         */
        inline constexpr uint64_t getSize() 
        { 
            return (static_cast<uint64_t>(_extraHeight) << 40) |
                   (static_cast<uint64_t>(_extraWidth) << 32) |
                   (static_cast<uint64_t>(_baseHeight) << 16) | 
                   _baseWidth; 
        }

        /**
         * @brief Sets the width of the original uncompressed image.
         */
        inline constexpr void setWidth(uint32_t width) 
        { 
            _baseWidth = width; 
            _extraWidth = width >> 16;
        }

        /**
         * @brief Gets the width of the original uncompressed image.
         */
        inline constexpr uint32_t getWidth() 
        { 
            return (static_cast<uint64_t>(_extraWidth) << 16) | _baseWidth; 
        }

        /**
         * @brief Sets the height of the original uncompressed image.
         */
        inline constexpr void setHeight(uint32_t height) 
        { 
            _baseHeight = height; 
            _extraHeight = height >> 16;
        }

        /**
         * @brief Gets the height of the original uncompressed image.
         */
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

    /**
     * @brief Base header for multi-threaded compression.
     */
    struct MultiThreadedHeader : public BasicHeader
    {
    public:
        /**
         * @brief Gets the number of threads used for multi-threaded compression.
         */
        inline constexpr uint8_t getNumberOfThreads() const { return _numberOfThreads; }

        /**
         * @brief Sets the number of threads used for multi-threaded compression.
         */
        inline constexpr void setNumberOfThreads(uint8_t numberOfThreads) { _numberOfThreads = numberOfThreads; }

        /**
         * @brief Sets the number of bits used for storing the size of compressed data by each thread.
         */
        inline constexpr void setBitsPerBlockSize(uint8_t bitsPerBlockSize) { _bitsPerBlockSize = bitsPerBlockSize;}

        /**
         * @brief Gets the number of bits used for storing the size of compressed data by each thread.
         */
        inline constexpr uint8_t getBitsPerBlockSize() { return _bitsPerBlockSize; }
    
    private:
        uint8_t _numberOfThreads;
        uint8_t _bitsPerBlockSize;
    } __attribute__((packed));

    /**
     * @brief Final header used for single-threaded compression, where the Huffman tree is stored as a depth bitmaps. 
     */
    struct DepthBitmapsHeader : public BasicHeader
    {
    public:
        inline constexpr uint16_t getCodeDepths() const { return _codeDepths; }
        inline constexpr void setCodeDepths(uint16_t codeDepths) { _codeDepths = codeDepths; }

    private:
        uint16_t _codeDepths;
    } __attribute__((packed));

    /**
     * @brief Final header used for multi-threaded compression, where the Huffman tree is stored as a depth bitmaps.
     *        Compressed depth bitmaps follow after the header.
     */
    struct DepthBitmapsMultiThreadedHeader : public MultiThreadedHeader
    {
    public:
        /**
         * @brief Gets a bitmap of used depths in the final huffman tree to decompress the following depth bitmaps.
         */
        inline constexpr uint16_t getCodeDepths() const { return _codeDepths; }

        /**
         * @brief Sets a bitmap of used depths in the final huffman tree to be able to decompress the following depth bitmaps.
         */
        inline constexpr void setCodeDepths(uint16_t codeDepths) { _codeDepths = codeDepths; }

    private:
        uint16_t _codeDepths;
    } __attribute__((packed));

    /**
     * @brief Initializes memory and variables necessary to store information about block traversals during adaptive compression/decompression.
     */
    inline constexpr void initializeBlockTypes()
    {
        _blocksPerRow = (_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        _blocksPerColumn = (_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
        _blockCount = _blocksPerRow * _blocksPerColumn;
        _bestBlockTraversals = new AdaptiveTraversals[_blockCount + 4];
        reinterpret_cast<uint32_t *>(_bestBlockTraversals + _blockCount)[0] = 0; // clear the last 4 bytes (padding)
    }

    /**
     * @brief Transposes a block of symbols and serializes it row by row into a destination buffer.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be transposed and serialized in the source buffer.
     * @param blockColumn Starting column index of the block to be transposed and serialized in the source buffer.
     */
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

    /**
     * @brief Deserializes a block from contiguous memory and transposes it back to the original form.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be transposed and deserialized in the source buffer.
     * @param blockColumn Starting column index of the block to be transposed and deserialized in the source buffer.
     */
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

    /**
     * @brief Serializes a block of symbols row by row into a destination buffer.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be serialized in the source buffer.
     * @param blockColumn Starting column index of the block to be serialized in the source buffer.
     */
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

    /**
     * @brief Deserializes a block from contiguous memory.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be deserialized in the source buffer.
     * @param blockColumn Starting column index of the block to be deserialized in the source buffer.
     */
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