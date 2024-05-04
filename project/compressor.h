/* =======================================================================================================================================================
 * Project:         Huffman RLE compression and decompression
 * Author:          David Mihola (xmihol00)
 * E-mail:          xmihol00@stud.fit.vutbr.cz
 * Date:            12. 5. 2024
 * Description:     A parallel implementation of a compression algorithm based on Huffman coding and RLE transformation. 
 *                  This file contains the compressor definitions.
 * ======================================================================================================================================================= */

#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "common.h"

/**
 * @brief A class implementing the compression of an input file using Huffman coding and RLE transformation.
 */
class Compressor : public HuffmanRLECompression
{
public:
    /**
     * @param model Flag indicating if the compression is model-based.
     * @param adaptive Flag indicating if the compression is adaptive.
     * @param width Width of the image to be compressed.
     * @param numberOfThreads Number of threads used for multi-threaded compression.
     */
    Compressor(bool model, bool adaptive, uint64_t width, int32_t numberOfThreads);
    ~Compressor() = default;

    /**
     * @brief Compresses the input file and writes the compressed data to the output file.
     *        NOTE: this method can be called repeatedly with different input and output file names on the same instance of the Compressor class.
     * @param inputFileName Name of the input file.
     * @param outputFileName Name of the output file.
     */
    void compress(std::string inputFileName, std::string outputFileName);

private:
    /**
     * @brief Represents the frequency of a symbol in the input data, packed to 32 bits for SIMD processing.
     */
    struct FrequencySymbolIndex
    {
        uint8_t index;                  ///< LSB: Index of the symbol, i.e. its value.
        uint8_t frequencyLowBits;       ///< Lower 8 bits of the frequency of the symbol.
        uint16_t frequencyHighBits;     ///< MSB: Higher 16 bits of the frequency of the symbol.
    } __attribute__((packed));

    /**
     * @brief Represents the parent and depth of a symbol in the Huffman tree.
     */
    struct SymbolParentDepth
    {
        uint8_t symbol;     ///< value of the symbol.
        uint8_t depth;      ///< Depth of the symbol in the Huffman tree.
        uint16_t parent;    ///< Index of the parent of the symbol.
    } __attribute__((packed));

    /**
     * @brief Represents a symbol in the Huffman code table.
     */
    struct HuffmanCode
    {
        uint16_t code;      ///< Huffman code of the symbol.
        uint16_t length;    ///< Length of the Huffman code.
    } __attribute__((packed));

    /**
     * @brief Symbol with highest frequency (UINT32_MAX) to represent unused symbols or already processed symbols.
     */
    static constexpr FrequencySymbolIndex EMPTY_FREQUENCY_SYMBOL_INDEX{ .index = 0xff, .frequencyLowBits = 0xff, .frequencyHighBits = 0xffff };

    /**
     * @brief Highest valid symbol frequency, all higher frequencies will be clamped to it (last significant byte is reserved for the index).
     */
    static constexpr uint32_t MAX_VALID_FREQUENCY_SYMBOL_INDEX{0xfffe'00};

    /**
     * @brief Maximum number of bits used for frequency representation.
     */
    static constexpr uint8_t MAX_BITS_FOR_FREQUENCY{24};

    /**
     * @brief Maximum number of threads used for histogram computation.
     */
    static constexpr uint8_t MAX_HISTOGRAM_THREADS{MAX_NUMBER_OF_THREADS};

    /**
     * @brief Frees all memory allocated for compression.
     */
    void freeMemory();

    /**
     * @brief Reads the input file into a memory buffer and computes its height based on the file size and width passed by the user.
     *        Checks weather the input file is rectangular.
     */
    void readInputFile();

    /**
     * @brief Computes a histogram of the input data with repetitions considered.
     */
    void computeHistogram();

    /**
     * @brief Build a Huffman tree based on frequencies of symbols in the computed histogram.
     */
    void buildHuffmanTree();

    /**
     * @brief Populates the code table with canonical Huffman codes based on depths of symbols in the Huffman tree.
     */
    void populateCodeTable();

    /**
     * @brief Transforms the input data considering repetitions (RLE) and encodes them with the Huffman codes.
     */
    void transformRLE(symbol_t *sourceData, uint16_t *compressedData, uint64_t &compressedSize, uint64_t &startingIdx);

    /**
     * @brief Creates a header for the compressed file.
     */
    void createHeader();

    /**
     * @brief Packs the compressed sizes of each thread into a number of bits determined by the larges size.
     */
    void packCompressedSizes(uint16_t &maxBitsCompressedSizes);

    /**
     * @brief Creates a depth map out of symbols and their depths in the Huffman tree.
     */
    void compressDepthMaps();

    /**
     * @brief Writes the compressed data with a header to the output file.
     */
    void writeOutputFile();

    /**
     * @brief Performs static compression.
     */
    void compressStatic();

    /**
     * @brief Performs adaptive compression.
     */
    void compressAdaptive();

    /**
     * @brief Performs static compression with model-based approach.
     */
    void compressStaticModel();

    /**
     * @brief Performs adaptive compression with model-based approach.
     */
    void compressAdaptiveModel();

    /**
     * @brief Analyzes the image to find the best adaptive approach for each block.
     */
    void analyzeImageAdaptive();

    /**
     * @brief Applies a difference model to the source data.
     */
    void applyDiferenceModel(symbol_t *source, symbol_t *destination);

    /**
     * @brief Serializes a block of symbols row by row into a destination buffer.
     * @param source Source buffer.
     * @param destination Destination buffer.
     * @param blockRow Starting row index of the block to be serialized in the source buffer.
     * @param blockColumn Starting column index of the block to be serialized in the source buffer.
     * @param rowIndices Ordered indices of rows in the block to be serialized.
     * @param colIndices Ordered indices of columns in the block to be serialized.
     */
    inline constexpr void serializeBlock(symbol_t *source, symbol_t *destination, uint32_t blockRow, uint32_t blockColumn, 
                                         const uint8_t rowIndices[BLOCK_SIZE * BLOCK_SIZE], const uint8_t colIndices[BLOCK_SIZE * BLOCK_SIZE])
    {
        uint64_t blockFirstIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE;
        uint64_t destinationIdx = blockRow * BLOCK_SIZE * _width + blockColumn * BLOCK_SIZE * BLOCK_SIZE;

        for (uint32_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
        {
            uint64_t valueIdx = blockFirstIdx + rowIndices[k] * _width + colIndices[k];
            destination[destinationIdx++] = source[valueIdx];
        }
    }

    inline constexpr void countRepetitions(AdaptiveTraversals traversal, 
                                           const uint8_t rowIndices[BLOCK_SIZE * BLOCK_SIZE], const uint8_t colIndices[BLOCK_SIZE * BLOCK_SIZE])
    {
        int16_t *rleCounts = _rlePerBlockCounts + traversal * _numberOfTraversalBlocks;

        int32_t blockIdx = 0;
        for (uint32_t i = 0; i < _height; i += BLOCK_SIZE)
        {
            uint64_t blockRowIdx = i * _width;
            for (uint32_t j = 0; j < _width; j += BLOCK_SIZE)
            {
                uint64_t blockFirstIdx = blockRowIdx + j;
                int32_t sameSymbolCount = -3;
                int32_t rleCount = 0;
                uint16_t lastSymbol = -1;

                for (uint32_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
                {
                    uint64_t valueIdx = blockFirstIdx + rowIndices[k] * _width + colIndices[k];
                    bool sameSymbol = _sourceBuffer[valueIdx] == lastSymbol;
                    lastSymbol = _sourceBuffer[valueIdx];
                    rleCount += !sameSymbol && sameSymbolCount >= -1 ? sameSymbolCount : 0;
                    sameSymbolCount++;
                    sameSymbolCount = sameSymbol ? sameSymbolCount : -3;
                }
                rleCounts[blockIdx++] = rleCount;
            }
        }
    }

    /**
     * @brief A memory pool recycled for different purposes during the compression.
     */
    uint8_t _memoryPool[(MAX_HISTOGRAM_THREADS + 2) * NUMBER_OF_SYMBOLS * sizeof(uint64_t)] __attribute__((aligned(64)));
    FrequencySymbolIndex *_structHistogram{reinterpret_cast<FrequencySymbolIndex *>(_memoryPool)};
    uint32v16_t *_vectorHistogram{reinterpret_cast<uint32v16_t *>(_memoryPool)};
    uint64_t *_intHistogram{reinterpret_cast<uint64_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
    SymbolParentDepth *_symbolsParentsDepths{reinterpret_cast<SymbolParentDepth *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
    uint16_t *_parentsSortedIndices{reinterpret_cast<uint16_t *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
    uint16_t * _symbolsPerDepth{reinterpret_cast<uint16_t *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(uint64_t) + NUMBER_OF_SYMBOLS * sizeof(uint16_t))};
    HuffmanCode *_codeTable{reinterpret_cast<HuffmanCode *>(_memoryPool)};
    symbol_t *_symbols{reinterpret_cast<symbol_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode))};
    uint8_t *_depths{reinterpret_cast<uint8_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode) + NUMBER_OF_SYMBOLS * sizeof(symbol_t))};
    uint8_t *_adjustedDepths{reinterpret_cast<uint8_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(HuffmanCode) + 2 * NUMBER_OF_SYMBOLS * sizeof(symbol_t))};
    uint8_t *_compressedDepthMaps{reinterpret_cast<uint8_t *>(_memoryPool)};
    
    bool _compressionUnsuccessful{false};   ///< Flag indicating if the compression was unsuccessful, i.e. no compression was achieved.
    uint16_t _numberOfSymbols;              ///< Number of symbols in the input data, not all 256 possible values must be present.

    uint8_t *_sourceBuffer{nullptr};        ///< Buffer for the input uncompressed data.
    uint8_t *_destinationBuffer{nullptr};   ///< Buffer for the output compressed data.

    uint16_t _headerSize;                   ///< Size of the final header in bytes.
    uint16_t _threadBlocksSizesSize{0};     ///< Sizes of the compressed blocks for each thread.
    uint64_t _threadPadding;                ///< Padding between the compressed blocks for each thread, if the compressed size of any thread is larger than the uncompressed size.

    int16_t *_rlePerBlockCounts{nullptr, }; ///< Counts of repetitions of symbols in each block when adaptive compression is used.

    uint8_t _mostPopulatedDepth;        ///< Depth with the most symbols in the Huffman tree.
    uint8_t _mostPopulatedDepthIdx;     ///< Index of the depth with the most symbols in the Huffman tree in an array of depths.
    uint16_t _maxSymbolsPerDepth;       ///< Maximum number of symbols in a single depth in the Huffman tree.
    uint16_t _compressedDepthMapsSize;  ///< Size of the compressed depth maps in bytes.

    uint8_t *_blockTypes{nullptr};      ///< Dynamically allocated array with packed best traversal options for each block during adaptive compression.
    uint32_t _blockTypesByteSize{0};    ///< Size of the packed block types array in bytes.

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

    std::chrono::_V2::system_clock::time_point _histogramComputationStart;
    inline void startHistogramComputationTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _histogramComputationStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopHistogramComputationTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point histogramComputationEnd = high_resolution_clock::now();
            _performanceCounters["Histogram computation time"] += duration_cast<microseconds>(histogramComputationEnd - _histogramComputationStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _huffmanTreeBuildStart;
    inline void startHuffmanTreeBuildTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        _huffmanTreeBuildStart = high_resolution_clock::now();
    #endif
    }
    inline void stopHuffmanTreeBuildTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point huffmanTreeBuildEnd = high_resolution_clock::now();
        _performanceCounters["Huffman tree build time"] += duration_cast<microseconds>(huffmanTreeBuildEnd - _huffmanTreeBuildStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _codeTablePopulationStart;
    inline void startCodeTablePopulationTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        _codeTablePopulationStart = high_resolution_clock::now();
    #endif
    }
    inline void stopCodeTablePopulationTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point codeTablePopulationEnd = high_resolution_clock::now();
        _performanceCounters["Code table population time"] += duration_cast<microseconds>(codeTablePopulationEnd - _codeTablePopulationStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _transformRLEStart;
    inline void startTransformRLETimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
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
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point transformRLEEnd = high_resolution_clock::now();
            _performanceCounters["RLE transform time"] += duration_cast<microseconds>(transformRLEEnd - _transformRLEStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _analyzeImageAdaptiveStart;
    inline void startAnalyzeImageAdaptiveTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _analyzeImageAdaptiveStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopAnalyzeImageAdaptiveTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point analyzeImageAdaptiveEnd = high_resolution_clock::now();
            _performanceCounters["Adaptive analysis time"] += duration_cast<microseconds>(analyzeImageAdaptiveEnd - _analyzeImageAdaptiveStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _serializeTraversalStart;
    inline void startSerializeTraversalTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp master
        {
            _serializeTraversalStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopSerializeTraversalTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp master
        {
            system_clock::time_point serializeTraversalEnd = high_resolution_clock::now();
            _performanceCounters["Serialize traversal time"] += duration_cast<microseconds>(serializeTraversalEnd - _serializeTraversalStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _applyDiferenceModelStart;
    inline void startApplyDiferenceModelTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _applyDiferenceModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopApplyDiferenceModelTimer()
    {
    #ifdef _MEASURE_PARTIAL_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point applyDiferenceModelEnd = high_resolution_clock::now();
            _performanceCounters["Difference model application time"] += duration_cast<microseconds>(applyDiferenceModelEnd - _applyDiferenceModelStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _staticCompressionStart;
    inline void startStaticCompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _staticCompressionStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopStaticCompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point staticCompressionEnd = high_resolution_clock::now();
            _performanceCounters["Static compression time"] += duration_cast<microseconds>(staticCompressionEnd - _staticCompressionStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _staticCompressionWithModelStart;
    inline void startStaticCompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _staticCompressionWithModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopStaticCompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point staticCompressionWithModelEnd = high_resolution_clock::now();
            _performanceCounters["Static compression with model time"] += duration_cast<microseconds>(staticCompressionWithModelEnd - _staticCompressionWithModelStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _adaptiveCompressionStart;
    inline void startAdaptiveCompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _adaptiveCompressionStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopAdaptiveCompressionTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point adaptiveCompressionEnd = high_resolution_clock::now();
            _performanceCounters["Adaptive compression time"] += duration_cast<microseconds>(adaptiveCompressionEnd - _adaptiveCompressionStart).count();
        }
    #endif
    }

    std::chrono::_V2::system_clock::time_point _adaptiveCompressionWithModelStart;
    inline void startAdaptiveCompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        #pragma omp barrier
        #pragma omp master
        {
            _adaptiveCompressionWithModelStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopAdaptiveCompressionWithModelTimer()
    {
    #ifdef _MEASURE_ALGORITHM_EXECUTION_TIMES_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        #pragma omp barrier
        #pragma omp master
        {
            system_clock::time_point adaptiveCompressionWithModelEnd = high_resolution_clock::now();
            _performanceCounters["Adaptive compression with model time"] += duration_cast<microseconds>(adaptiveCompressionWithModelEnd - _adaptiveCompressionWithModelStart).count();
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

    uint64_t _traversalTypeCounts[NUMBER_OF_TRAVERSALS] = {0, };
    inline void countTraversalTypes()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        #pragma omp single
        {
            for (uint64_t i = 0; i < _numberOfTraversalBlocks; i++)
            {
                _traversalTypeCounts[_bestBlockTraversals[i]]++;
            }

            _performanceCounters["Horizontal zig-zag traversals"] = _traversalTypeCounts[HORIZONTAL_ZIG_ZAG];
            _performanceCounters["Vertical zig-zag traversals"] = _traversalTypeCounts[VERTICAL_ZIG_ZAG];
            _performanceCounters["Major diagonal zig-zag traversals"] = _traversalTypeCounts[MAJOR_DIAGONAL_ZIG_ZAG];
            _performanceCounters["Minor diagonal zig-zag traversals"] = _traversalTypeCounts[MINOR_DIAGONAL_ZIG_ZAG];
        }
    #endif
    }

    inline void captureHeaderSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Header size"] = _headerSize;
    #endif
    }

    inline void captureThreadBlocksSizesSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Thread blocks sizes size"] = _threadBlocksSizesSize;
    #endif
    }

    inline void captureCompressedDepthMapsSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Compressed depth maps size"] = _compressedDepthMapsSize + 2;
    #endif
    }

    inline void captureBlockTypesByteSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Block types byte size"] = _blockTypesByteSize;
    #endif
    }

    inline void captureCompressedDataSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Compressed data size"] = _compressedSizesExScan[_numberOfThreads];
    #endif
    }

    inline void captureWholeCompressedFileSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Compressed file size"] = _headerSize + _threadBlocksSizesSize + _compressedDepthMapsSize + _blockTypesByteSize + _compressedSizesExScan[_numberOfThreads];
    #endif
    }

    inline void captureUncompressedFileSize()
    {
    #ifdef _PERFORM_DATA_ANALYSIS_
        _performanceCounters["Uncompressed file size"] = _size;
    #endif
    }
};

#endif