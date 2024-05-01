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

#define _MEASURE_ALL_ 0

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
    ~Compressor();

    /**
     * @brief Compresses the input file and writes the compressed data to the output file.
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
    static constexpr FrequencySymbolIndex EMPTY_FREQUENCY_SYMBOL_INDEX = { .index = 0xff, .frequencyLowBits = 0xff, .frequencyHighBits = 0xffff };

    /**
     * @brief Highest valid symbol frequency, all higher frequencies will be clamped to it (last significant byte is reserved for the index).
     */
    static constexpr uint32_t MAX_VALID_FREQUENCY_SYMBOL_INDEX = 0xfffe'00;

    /**
     * @brief Maximum number of bits used for frequency representation.
     */
    static constexpr uint8_t MAX_BITS_FOR_FREQUENCY = 24;

    /**
     * @brief Maximum number of threads used for histogram computation.
     */
    static constexpr uint8_t MAX_HISTOGRAM_THREADS = 8;

    /**
     * @brief Reads the input file into a memory buffer and computes its height based on the file size and width passed by the user.
     *        Checks weather the input file is rectangular.
     * @param inputFileName Name of the input file.
     * @param outputFileName Name of the output file.
     */
    void readInputFile(std::string inputFileName, std::string outputFileName);

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
     * @brief Writes the compressed data with a header to the output file
     */
    void writeOutputFile(std::string outputFileName, std::string inputFileName);

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

        #pragma GCC unroll BLOCK_SIZE * BLOCK_SIZE
        for (uint32_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
        {
            uint64_t valueIdx = blockFirstIdx + rowIndices[k] * _width + colIndices[k];
            destination[destinationIdx++] = source[valueIdx];
        }
    }

    inline constexpr void countRepetitions(AdaptiveTraversals traversal, 
                                           const uint8_t rowIndices[BLOCK_SIZE * BLOCK_SIZE], const uint8_t colIndices[BLOCK_SIZE * BLOCK_SIZE])
    {
        _rlePerBlockCounts[traversal] = new int16_t[_numberOfTraversalBlocks];
        int16_t *rleCounts = _rlePerBlockCounts[traversal];

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

                #pragma GCC unroll BLOCK_SIZE * BLOCK_SIZE
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
    uint8_t _memoryPool[10 * NUMBER_OF_SYMBOLS * sizeof(uint64_t)] __attribute__((aligned(64)));
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

    int16_t *_rlePerBlockCounts[NUMBER_OF_TRAVERSALS]{nullptr, }; ///< Counts of repetitions of symbols in each block when adaptive compression is used.

    uint8_t _mostPopulatedDepth;        ///< Depth with the most symbols in the Huffman tree.
    uint8_t _mostPopulatedDepthIdx;     ///< Index of the depth with the most symbols in the Huffman tree in an array of depths.
    uint16_t _maxSymbolsPerDepth;       ///< Maximum number of symbols in a single depth in the Huffman tree.
    uint16_t _compressedDepthMapsSize;  ///< Size of the compressed depth maps in bytes.

    // ------------------------------------------------------------------------------------------
    // Section for performance measurements
    // ------------------------------------------------------------------------------------------

    std::chrono::_V2::system_clock::time_point _readInputFileStart;
    inline void startReadInputFileTimer()
    {
    #if _MEASURE_READ_INPUT_FILE_ || _MEASURE_ALL_
        using namespace std::chrono;
        _readInputFileStart = high_resolution_clock::now();
    #endif
    }
    inline void stopReadInputFileTimer()
    {
    #if _MEASURE_READ_INPUT_FILE_ || _MEASURE_ALL_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point readInputFileEnd = high_resolution_clock::now();
        _performanceCounters["Read input file time"] += duration_cast<microseconds>(readInputFileEnd - _readInputFileStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _writeOutputFileStart;
    inline void startWriteOutputFileTimer()
    {
    #if _MEASURE_WRITE_OUTPUT_FILE_ || _MEASURE_ALL_
        using namespace std::chrono;
        _writeOutputFileStart = high_resolution_clock::now();
    #endif
    }
    inline void stopWriteOutputFileTimer()
    {
    #if _MEASURE_WRITE_OUTPUT_FILE_ || _MEASURE_ALL_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point writeOutputFileEnd = high_resolution_clock::now();
        _performanceCounters["Write output file time"] += duration_cast<microseconds>(writeOutputFileEnd - _writeOutputFileStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _histogramComputationStart;
    inline void startHistogramComputationTimer()
    {
    #if _MEASURE_HISTOGRAM_ || _MEASURE_ALL_
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
    #if _MEASURE_HISTOGRAM_ || _MEASURE_ALL_
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
    #if _MEASURE_HUFFMAN_TREE_BUILD_ || _MEASURE_ALL_
        using namespace std::chrono;
        _huffmanTreeBuildStart = high_resolution_clock::now();
    #endif
    }
    inline void stopHuffmanTreeBuildTimer()
    {
    #if _MEASURE_HUFFMAN_TREE_BUILD_ || _MEASURE_ALL_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point huffmanTreeBuildEnd = high_resolution_clock::now();
        _performanceCounters["Huffman tree build time"] += duration_cast<microseconds>(huffmanTreeBuildEnd - _huffmanTreeBuildStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _codeTablePopulationStart;
    inline void startCodeTablePopulationTimer()
    {
    #if _MEASURE_CODE_TABLE_POPULATION_ || _MEASURE_ALL_
        using namespace std::chrono;
        _codeTablePopulationStart = high_resolution_clock::now();
    #endif
    }
    inline void stopCodeTablePopulationTimer()
    {
    #if _MEASURE_CODE_TABLE_POPULATION_ || _MEASURE_ALL_
        using namespace std::chrono;
        using namespace std::chrono::_V2;
        system_clock::time_point codeTablePopulationEnd = high_resolution_clock::now();
        _performanceCounters["Code table population time"] += duration_cast<microseconds>(codeTablePopulationEnd - _codeTablePopulationStart).count();
    #endif
    }

    std::chrono::_V2::system_clock::time_point _transformRLEStart;
    inline void startTransformRLETimer()
    {
    #if _MEASURE_RLE_TRANSFORM_ || _MEASURE_ALL_
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
    #if _MEASURE_RLE_TRANSFORM_ || _MEASURE_ALL_
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
    #if _MEASURE_ADAPTIVE_ANALYSIS_ || _MEASURE_ALL_
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
    #if _MEASURE_ADAPTIVE_ANALYSIS_ || _MEASURE_ALL_
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
    #if _MEASURE_SERIALIZE_TRAVERSAL_ || _MEASURE_ALL_
        using namespace std::chrono;
        #pragma omp master
        {
            _serializeTraversalStart = high_resolution_clock::now();
        }
    #endif
    }
    inline void stopSerializeTraversalTimer()
    {
    #if _MEASURE_SERIALIZE_TRAVERSAL_ || _MEASURE_ALL_
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
    #if _MEASURE_DIFFERENCE_MODEL_ || _MEASURE_ALL_
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
    #if _MEASURE_DIFFERENCE_MODEL_ || _MEASURE_ALL_
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
    #if _MEASURE_STATIC_COMPRESSION_
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
    #if _MEASURE_STATIC_COMPRESSION_
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
    #if _MEASURE_STATIC_COMPRESSION_WITH_MODEL_
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
    #if _MEASURE_STATIC_COMPRESSION_WITH_MODEL_
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
    #if _MEASURE_ADAPTIVE_COMPRESSION_
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
    #if _MEASURE_ADAPTIVE_COMPRESSION_
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
    #if _MEASURE_ADAPTIVE_COMPRESSION_WITH_MODEL_
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
    #if _MEASURE_ADAPTIVE_COMPRESSION_WITH_MODEL_ || _MEASURE_ALL_
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

    uint64_t _traversalTypeCounts[NUMBER_OF_TRAVERSALS] = {0, };
    inline void countTraversalTypes()
    {
    #if _MEASURE_TRAVERSAL_TYPES_ || _MEASURE_ALL_
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
};

#endif