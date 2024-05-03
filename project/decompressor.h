/* =======================================================================================================================================================
 * Project:         Huffman RLE compression and decompression
 * Author:          David Mihola (xmihol00)
 * E-mail:          xmihol00@stud.fit.vutbr.cz
 * Date:            12. 5. 2024
 * Description:     A parallel implementation of a compression algorithm based on Huffman coding and RLE transformation. 
 *                  This file contains the decompressor definitions.
 * ======================================================================================================================================================= */

#ifndef _DECOMPRESSOR_H_
#define _DECOMPRESSOR_H_

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <bitset>

#include "common.h"

/**
 * @brief A class implementing the decompression of an input file compressed using Huffman coding and RLE transformation.
 */
class Decompressor : public HuffmanRLECompression
{
public:
    /**
     * @param numberOfThreads Number of threads used for multi-threaded decompression.
     */
    Decompressor(int32_t numberOfThreads);
    ~Decompressor() = default;

    /**
     * @brief Decompresses the input file and writes the decompressed data to the output file.
     *        NOTE: All memory allocated for decompression is freed and the method can be called again without deleting the object.
     * @param inputFileName Name of the input file.
     * @param outputFileName Name of the output file.
     */
    void decompress(std::string inputFileName, std::string outputFileName);

private:
    /**
     * @brief Frees the memory allocated for decompression.
     */
    void freeMemory();

    /**
     * @brief Reads the input file and parses its header. If the header indicates that no compression was performed, 
     *        the rest of the file is copied to the output file.
     */
    bool readInputFile();

    /**
     * @brief Writes the decompressed data to the output file.
     */
    void writeOutputFile();

    /**
     * @brief Parses a depth bitmap of symbols in a Huffman tree from the input file.
     * @param readBytes Number of bytes read from the input file.
     */
    void parseHuffmanTree(uint16_t &readBytes);

    /**
     * @brief Parses the threading information from the input file.
     */
    void parseThreadingInfo();

    /**
     * @brief Decodes the compressed data using the Huffman tree and applies the inverse to the RLE transformation.
     * @param compressedData Buffer with compressed data.
     * @param decompressedData Buffer for decompressed data.
     * @param bytesToDecompress Number of bytes to decompress.
     */
    void transformRLE(uint16_t *compressedData, symbol_t *decompressedData, uint64_t bytesToDecompress);

    /**
     * @brief Applies the inverse of the difference model to the decompressed data.
     * @param source Buffer with decompressed data.
     * @param destination Buffer for the inverse differentiated decompressed data.
     */
    void reverseDifferenceModel(symbol_t *source, symbol_t *destination, uint64_t bytesToProcess);

    /**
     * @brief Decompresses the data compressed using a static compression.
     */
    void decompressStatic();

    /**
     * @brief Decompresses the data compressed using an adaptive compression.
     */
    void decompressAdaptive();

    /**
     * @brief Decompresses the data compressed using a static model-based compression.
     */
    void decompressStaticModel();

    /**
     * @brief Decompresses the data compressed using an adaptive model-based compression.
     */
    void decompressAdaptiveModel();

    /**
     * @brief Deserializes a block from contiguous memory.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be deserialized in the source buffer.
     * @param blockColumn Starting column index of the block to be deserialized in the source buffer.
     * @param rowIndices Ordered indices of rows in the block to be deserialized.
     * @param colIndices Ordered indices of columns in the block to be deserialized.
     */
    inline constexpr void deserializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn, 
                                           const uint8_t rowIndices[BLOCK_SIZE * BLOCK_SIZE], const uint8_t colIndices[BLOCK_SIZE * BLOCK_SIZE])
    {
        uint32_t sourceIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;
        uint32_t blockFirstIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;

        for (uint32_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
        {
            uint64_t valueIdx = blockFirstIdx + rowIndices[k] * _width + colIndices[k];
            destination[valueIdx] = source[sourceIdx++];
        }   
    }

    uint8_t _numberOfCompressedBlocks{1};    ///< Number of compressed blocks in the input file, which is equal to the number of threads used for compression.
    uint8_t _bitsPerCompressedBlockSize;     ///< Number of bits used to store the size of a each compressed block.
    uint32_t _numberOfBytesCompressedBlocks; ///< Total number of bytes used to store the sizes of compressed blocks.
    uint8_t *_rawBlockSizes{reinterpret_cast<uint8_t *>(_compressedSizesExScan)}; ///< Pointer alias to reuse memory for the block sizes processing.
    uint8_t _rawDepthBitmaps[MAX_NUMBER_OF_CODES * NUMBER_OF_SYMBOLS];  ///< Buffer for compressed depth bitmaps of symbols in a Huffman tree.

    uint8_t *_decompressionBuffer{nullptr};  ///< Buffer for decompressed data.
    uint16_t *_compressedData{nullptr};      ///< Buffer for compressed data.
    symbol_t *_decompressedData{nullptr};    ///< Alias representing the decompressed data at the end of the decompression process.

    FirstByteHeader &_header = reinterpret_cast<FirstByteHeader &>(_headerBuffer);  ///< Memory buffer used to store and process different headers.

    /**
     * @brief Represent a dictionary between prefix lengths of a Huffman code and its index in an array of vectors of prefixes and masks. 
     */
    struct PrefixIndexLength
    {
        uint16_t index;     ///< The index to the array of vectors of prefixes and masks
        int16_t length;     ///< The prefix length of the associated Huffman code.
    } _prefixIndexLengths[MAX_NUMBER_OF_PREFIXES] __attribute__((aligned(64)));

    /**
     * @brief Represents a batch of symbols encoded with the same prefix in the same depth of a Huffman tree.
     */
    struct DepthIndices
    {
        uint8_t symbolsAtDepthIndex;  ///< Index to an array of symbol depths maps in the Huffman tree.
        uint8_t masksIndex;           ///< Index to an array of vectors of prefixes and masks.
        uint8_t depth;                ///< The depth of the associated symbols.
        uint8_t prefixLength;         ///< The shared prefix length among the symbols.
    } _depthsIndices[MAX_NUMBER_OF_PREFIXES] __attribute__((aligned(64)));

    /**
     * @brief Packs an offset index of a specific prefix to a symbol table, shift necessary to remove the prefix, 
     *        shift necessary to align the suffix to LSB and the whole length of the Huffman code.
     */
    struct IndexShiftsCodeLength
    {
        uint8_t prefixIndex; ///< Offset index to a symbol table shared between symbols with the same prefix.
        uint8_t prefixShift; ///< Number of bits necessary to shift out the prefix.
        uint8_t suffixShift; ///< Number of bits necessary to align the suffix with LSB of a uint16_t, must be applied after the prefix shift.
        uint8_t codeLength;  ///< The overall length of the Huffman codes with the given prefix.
    } _indexShiftsCodeLengths[MAX_NUMBER_OF_PREFIXES] __attribute__((aligned(64)));

    uint8_t _symbolsTable[NUMBER_OF_SYMBOLS * 2] __attribute__((aligned(64)));   ///< Table with values of decompressed symbols.

    uint16v32_t _codePrefixesVector __attribute__((aligned(64)));                ///< Vector of prefixes of Huffman codes.
    uint16_t *_codePrefixes{reinterpret_cast<uint16_t *>(&_codePrefixesVector)}; ///< Array of prefixes for easier access to the vector.
    uint16v32_t _codeMasksVector;                                                ///< Vector of masks for the corresponding prefixes.
    uint16_t *_codeMasks{reinterpret_cast<uint16_t *>(&_codeMasksVector)};       ///< Array of masks for easier access to the vector.

    // ------------------------------------------------------------------------------------------
    // Section for performance measurements
    // ------------------------------------------------------------------------------------------

    std::chrono::_V2::system_clock::time_point _readInputFileStart;
    inline void startReadInputFileTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        _readInputFileStart = high_resolution_clock::now();
    #endif
    }
    inline void stopReadInputFileTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point readInputFileEnd = high_resolution_clock::now();
        _performanceCounters["Read input file time"] += duration_cast<microseconds>(readInputFileEnd - _readInputFileStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _writeOutputFileStart;
    inline void startWriteOutputFileTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        _writeOutputFileStart = high_resolution_clock::now();
    #endif
    }
    inline void stopWriteOutputFileTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point writeOutputFileEnd = high_resolution_clock::now();
        _performanceCounters["Write output file time"] += duration_cast<microseconds>(writeOutputFileEnd - _writeOutputFileStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _huffmanTreeRebuildStart;
    inline void startHuffmanTreeRebuildTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        _huffmanTreeRebuildStart = high_resolution_clock::now();
    #endif
    }
    inline void stopHuffmanTreeRebuildTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point huffmanTreeRebuildEnd = high_resolution_clock::now();
        _performanceCounters["Huffman tree rebuild time"] += duration_cast<microseconds>(huffmanTreeRebuildEnd - _huffmanTreeRebuildStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _transformRLEStart;
    inline void startTransformRLETimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp master
        {
            _transformRLEStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopTransformRLETimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point transformRLEEnd = high_resolution_clock::now();
            _performanceCounters["RLE transform time"] += duration_cast<microseconds>(transformRLEEnd - _transformRLEStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _deserializeTraversalStart;
    inline void startDeserializeTraversalTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp master
        {
            _deserializeTraversalStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopDeserializeTraversalTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point deserializeTraversalEnd = high_resolution_clock::now();
            _performanceCounters["Deserialize traversal time"] += duration_cast<microseconds>(deserializeTraversalEnd - _deserializeTraversalStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _reverseDiferenceModelStart;
    inline void startReverseDiferenceModelTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp master
        {
            _reverseDiferenceModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopReverseDiferenceModelTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point applyDiferenceModelEnd = high_resolution_clock::now();
            _performanceCounters["Difference model reversal time"] += duration_cast<microseconds>(applyDiferenceModelEnd - _reverseDiferenceModelStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _staticDecompressionStart;
    inline void startStaticDecompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _staticDecompressionStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopStaticDecompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point staticDecompressionEnd = high_resolution_clock::now();
            _performanceCounters["Static decompression time"] += duration_cast<microseconds>(staticDecompressionEnd - _staticDecompressionStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _staticDecompressionWithModelStart;
    inline void startStaticDecompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _staticDecompressionWithModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopStaticDecompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point staticDecompressionWithModelEnd = high_resolution_clock::now();
            _performanceCounters["Static decompression with model time"] += duration_cast<microseconds>(staticDecompressionWithModelEnd - _staticDecompressionWithModelStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _adaptiveDecompressionStart;
    inline void startAdaptiveDecompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _adaptiveDecompressionStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopAdaptiveDecompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point adaptiveDecompressionEnd = high_resolution_clock::now();
            _performanceCounters["Adaptive decompression time"] += duration_cast<microseconds>(adaptiveDecompressionEnd - _adaptiveDecompressionStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _adaptiveDecompressionWithModelStart;
    inline void startAdaptiveDecompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _adaptiveDecompressionWithModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopAdaptiveDecompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point adaptiveDecompressionWithModelEnd = high_resolution_clock::now();
            _performanceCounters["Adaptive decompression with model time"] += duration_cast<microseconds>(adaptiveDecompressionWithModelEnd - _adaptiveDecompressionWithModelStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _fullExecutionStart;
    inline void startFullExecutionTimer()
    {
    #ifdef _MEASURE_FULL_EXECUTION_TIME_
        using namespace std::chrono;
        _fullExecutionStart = high_resolution_clock::now();
    #endif
    }
    inline void stopFullExecutionTimer()
    {
    #ifdef _MEASURE_FULL_EXECUTION_TIME_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point fullExecutionEnd = high_resolution_clock::now();
        _performanceCounters["Full execution time"] = duration_cast<microseconds>(fullExecutionEnd - _fullExecutionStart).count();
    #endif
    }
};

#endif // _DECOMPRESSOR_H_