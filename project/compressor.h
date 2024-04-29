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

    static constexpr FrequencySymbolIndex MAX_FREQUENCY_SYMBOL_INDEX = { .index = 0xff, .frequencyLowBits = 0xff, .frequencyHighBits = 0xffff };
    /**
     * @brief Maximum number of bits used for frequency representation.
     */

    static constexpr uint8_t MAX_BITS_FOR_FREQUENCY = 16;
    /**
     * @brief Maximum number of threads used for histogram computation.
     */
    static constexpr uint8_t MAX_HISTOGRAM_THREADS = 8;

    /**
     * @brief Reads the input file into a memory buffer and computes its height based on the file size and width passed by the user.
     *        Checks weather the input file is rectangular.
     */
    void readInputFile(std::string inputFileName);

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
    void transformRLE(symbol_t *sourceData, uint16_t *compressedData, uint32_t &compressedSize, uint64_t &startingIdx);

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
     * @brief A memory pool recycled for different purposes during the compression.
     */
    uint8_t _memoryPool[10 * NUMBER_OF_SYMBOLS * sizeof(uint64_t)] __attribute__((aligned(64)));
    FrequencySymbolIndex *_structHistogram{reinterpret_cast<FrequencySymbolIndex *>(_memoryPool)};
    uint32v16_t *_vectorHistogram{reinterpret_cast<uint32v16_t *>(_memoryPool)};
    uint64_t *_intHistogram{reinterpret_cast<uint64_t *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
    SymbolParentDepth *_symbolsParentsDepths{reinterpret_cast<SymbolParentDepth *>(_memoryPool + NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
    uint16_t *_parentsSortedIndices{reinterpret_cast<uint16_t *>(_memoryPool + 3 * NUMBER_OF_SYMBOLS * sizeof(uint64_t))};
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

    int32_t *_rlePerBlockCounts[MAX_NUM_THREADS]{nullptr, }; ///< Counts of repetitions of symbols in each block when adaptive compression is used.

    uint8_t _mostPopulatedDepth;        ///< Depth with the most symbols in the Huffman tree.
    uint8_t _mostPopulatedDepthIdx;     ///< Index of the depth with the most symbols in the Huffman tree in an array of depths.
    uint16_t _maxSymbolsPerDepth;       ///< Maximum number of symbols in a single depth in the Huffman tree.
    uint16_t _compressedDepthMapsSize;  ///< Size of the compressed depth maps in bytes.
};

#endif