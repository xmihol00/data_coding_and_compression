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

#include <unordered_map>
#include <chrono>

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
     * @brief Exit codes of the compressor/decompressor.
     */
    enum ExitCodes
    {
        SUCCESS = 0,
        INVALID_ARGUMENT = 2,
        INPUT_FILE_ERROR = 4,
        OUTPUT_FILE_ERROR = 8,
        FILE_SIZE_ERROR = 16,
        MEMORY_ALLOCATION_ERROR = 32,
        CORRUPTED_FILE_ERROR = 64,
    };

    /**
     * @brief Maximum number of bits used to encode the width and height of the input file.
     */
    static constexpr uint16_t MAX_BITS_PER_FILE_DIMENSION{24};

    /**
     * @brief Maximum number of bits used to encode the size of the input file.
     */
    static constexpr uint16_t MAX_BITS_FOR_FILE_SIZE{48};

    /**
     * @param model Flag indicating if the compression is model-based, unused for decompression.
     * @param adaptive Flag indicating if the compression is adaptive, unused for decompression.
     * @param width Width of the image to be compressed, unused for decompression.
     * @param numberOfThreads Number of threads used for multi-threaded compression/decompression.
     */
    HuffmanRLECompression(bool model = false, bool adaptive = false, uint64_t width = 0, int32_t numberOfThreads = 1) 
        : _model{model}, _adaptive{adaptive}, _width{width}, _numberOfThreads{numberOfThreads} { };
    ~HuffmanRLECompression() = default;

protected:
    /**
     * @brief Types of image traversal for adaptive compression.
     */
    enum AdaptiveTraversals
    {
        HORIZONTAL_ZIG_ZAG = 0,
        VERTICAL_ZIG_ZAG = 1,
        MAJOR_DIAGONAL_ZIG_ZAG = 2,
        MINOR_DIAGONAL_ZIG_ZAG = 3,
        NUMBER_OF_TRAVERSALS
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
    static constexpr uint16_t MAX_NUMBER_OF_THREADS{32};

    /**
     * @brief Number of bits (size of a repetition chunk) used to encode repetitions of a character, the MSB is used to determine if the next chunk of bits is 
     *        the continuation of the number of repetitions (MSB is set), or the number of repetitions ends by this chunk (MSB is not set).
     */
    static constexpr uint16_t BITS_PER_REPETITION_NUMBER{4};

    /**
     * @brief Number of bits used to encode the type of a block (horizontal or vertical traversal) during adaptive compression.
     */
    static constexpr uint16_t BITS_PER_BLOCK_TYPE{2};

    /**
     * @brief Number of repetitions of the same symbol that has to be seen to encode the repetitions of the symbol.
     */
    static constexpr uint16_t SAME_SYMBOLS_TO_REPETITION{3};

    // 2D indices for various traversals of a 16x16 block of memory
    constexpr static uint8_t HORIZONTAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
    constexpr static uint8_t HORIZONTAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    constexpr static uint8_t VERTICAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    constexpr static uint8_t VERTICAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

    constexpr static uint8_t MAJOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 12, 13, 14, 15, 15, 14, 13, 14, 15, 15};
    constexpr static uint8_t MAJOR_DIAGONAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {15, 14, 15, 15, 14, 13, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 0};

    constexpr static uint8_t MINOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 12, 13, 14, 15, 15, 14, 13, 14, 15, 15};
    constexpr static uint8_t MINOR_DIAGONAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 13, 14, 15, 15, 14, 15};

    // Lookup tables for the 2D indices of the zig-zag traversals
    constexpr static uint8_t const *ROW_INDICES_LOOKUP_TABLE[4] = {
        HORIZONTAL_ZIG_ZAG_ROW_INDICES, VERTICAL_ZIG_ZAG_ROW_INDICES, MAJOR_DIAGONAL_ZIG_ZAG_ROW_INDICES, MINOR_DIAGONAL_ZIG_ZAG_ROW_INDICES
    };
    constexpr static uint8_t const *COL_INDICES_LOOKUP_TABLE[4] = {
        HORIZONTAL_ZIG_ZAG_COL_INDICES, VERTICAL_ZIG_ZAG_COL_INDICES, MAJOR_DIAGONAL_ZIG_ZAG_COL_INDICES, MINOR_DIAGONAL_ZIG_ZAG_COL_INDICES
    };

    bool _model;                  ///< Flag indicating if the compression is model-based.
    bool _adaptive;               ///< Flag indicating if the compression is adaptive.
    uint64_t _width;              ///< Width of the image to be compressed/decompressed.
    uint64_t _height;             ///< Height of the image to be compressed/decompressed.
    uint64_t _size;               ///< Overall size of the image to be compressed/decompressed.
    std::string _inputFileName;   ///< Name of the input file.
    std::string _outputFileName;  ///< Name of the output file.

    uint64_t _numberOfTraversalBlocks;       ///< Total number of blocks in an image.
    uint32_t _blocksPerRow;     ///< Number of blocks in a row of an image.
    uint32_t _blocksPerColumn;  ///< Number of blocks in a column of an image.

    int32_t _numberOfThreads;   ///< Number of active threads used for multi-threaded compression.

    uint64_t _compressedSizes[MAX_NUMBER_OF_THREADS + 1];         ///< Sizes of compressed data by each thread.
    uint64_t _compressedSizesExScan[MAX_NUMBER_OF_THREADS + 1];   ///< Exclusive scan of compressed sizes by each thread.

    uint8_t _symbolsPerDepth[MAX_NUMBER_OF_CODES * 2];      ///< Number of symbols at each depth of the huffman tree.
    uint32_t _usedDepths;                                   ///< Bitmap of used depths in the final huffman tree (rebalanced if needed).
    uint64v4_t _symbolsAtDepths[MAX_NUMBER_OF_CODES * 2] __attribute__((aligned(64))); ///< Number of symbols at each depth of the final huffman tree.

    uint8_t _headerBuffer[32];  ///< Memory buffer to store a specific header depending on the type of the compression and number of threads.

    AdaptiveTraversals *_bestBlockTraversals{nullptr}; ///< Dynamically allocated array with the best traversal option for each block during adaptive compression.
    
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
        _numberOfTraversalBlocks = _blocksPerRow * _blocksPerColumn;
        _bestBlockTraversals = new AdaptiveTraversals[_numberOfTraversalBlocks + 4];
        
        // clear the last 4 bytes (padding)
        _bestBlockTraversals[_numberOfTraversalBlocks] = HORIZONTAL_ZIG_ZAG;
        _bestBlockTraversals[_numberOfTraversalBlocks + 1] = HORIZONTAL_ZIG_ZAG;
        _bestBlockTraversals[_numberOfTraversalBlocks + 2] = HORIZONTAL_ZIG_ZAG;
        _bestBlockTraversals[_numberOfTraversalBlocks + 3] = HORIZONTAL_ZIG_ZAG;
    }

    // Section with variables for performance testing
    std::unordered_map<std::string, uint64_t> _performanceCounters; ///< Map with performance counters for different parts of the compression/decompression process.

    /**
     * @brief Prints all stored performance counters during the compression/decompression process.
     */
    inline void printPerformanceCounters()
    {
    #if _MEASURE_PARTIAL_EXECUTION_TIMES_ || _MEASURE_ALGORITHM_EXECUTION_TIMES_ || _MEASURE_FULL_EXECUTION_TIME_ || _PERFORM_DATA_ANALYSIS_
        for (const auto& [key, value] : _performanceCounters)
        {
            std::cout << _inputFileName << "," << key << "," << value << "," << _numberOfThreads << "," << _adaptive << "," << _model << std::endl;
        }
    #endif
    }
};

#endif // _COMMON_H_