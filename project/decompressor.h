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
    ~Decompressor();

    /**
     * @brief Decompresses the input file and writes the decompressed data to the output file.
     * @param inputFileName Name of the input file.
     * @param outputFileName Name of the output file.
     */
    void decompress(std::string inputFileName, std::string outputFileName);

private:
    /**
     * @brief Reads the input file and parses its header. If the header indicates that no compression was performed, 
     *        the rest of the file is copied to the output file.
     * @param inputFileName Name of the input file.
     * @param outputFileName Name of the output file.
     */
    bool readInputFile(std::string inputFileName, std::string outputFileName);

    /**
     * @brief Writes the decompressed data to the output file.
     * @param outputFileName Name of the output file.
     */
    void writeOutputFile(std::string outputFileName);

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
    struct IndexPrefixLength
    {
        uint16_t index;        ///< The index to the array of vectors of prefixes and masks  //TODO: rename
        int16_t prefixLength;  ///< The prefix length of the associated Huffman code.
    } _indexPrefixLengths[MAX_NUMBER_OF_PREFIXES] __attribute__((aligned(64)));

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
};

#endif // _DECOMPRESSOR_H_