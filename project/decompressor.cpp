/* =======================================================================================================================================================
 * Project:         Huffman RLE compression and decompression
 * Author:          David Mihola (xmihol00)
 * E-mail:          xmihol00@stud.fit.vutbr.cz
 * Date:            12. 5. 2024
 * Description:     A parallel implementation of a compression algorithm based on Huffman coding and RLE transformation. 
 *                  This file contains the implementation of the decompressor.
 * ======================================================================================================================================================= */

#include "decompressor.h"

using namespace std;

Decompressor::Decompressor(int32_t numberOfThreads)
    : HuffmanRLECompression(false, false, 0, numberOfThreads) { }

Decompressor::~Decompressor()
{
    if (_decompressionBuffer != nullptr)
    {
        free(_decompressionBuffer);
    }

    if (_compressedData != nullptr)
    {
        free(_compressedData);
    }
}

bool Decompressor::readInputFile(string inputFileName, string outputFileName)
{
    ifstream inputFile(inputFileName, ios::binary);
    if (!inputFile.is_open())
    {
        cout << "Error: Could not open input file " << inputFileName << endl;
        exit(1);
    }

    inputFile.seekg(0, ios::end);
    uint64_t fileSize = inputFile.tellg();
    inputFile.seekg(0, ios::beg);
    uint64_t alreadyReadBytes = 0;

    FirstByteHeader firstByte;
    inputFile.read(reinterpret_cast<char *>(&firstByte), sizeof(FirstByteHeader));
    if (firstByte.getCompressed()) // file is not compressed
    {
        ofstream outputFile(outputFileName, ios::binary);
        if (!outputFile.is_open())
        {
            cout << "Error: Could not open output file " << outputFileName << endl;
            exit(1);
        }

        // just copy it to the output file
        char *buffer = new char[fileSize];
        inputFile.read(buffer, fileSize - 1);
        outputFile.write(buffer, fileSize - 1);

        delete[] buffer;
        outputFile.close();
        inputFile.close();
        
        return false;
    }

    inputFile.seekg(0, ios::beg); // rewind the file
    uint16_t bitmapsSize;
    switch (firstByte.getHeaderType() & MULTI_THREADED) // load the correct header based on the header type in the first byte
    {
        case SINGLE_THREADED:
            {
                DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_headerBuffer); // use the preallocated buffer for the header
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsHeader));      // read the whole header (including the first byte again)
                
                // read the compressed depth bitmaps (the compressed Huffman tree)
                _usedDepths = header.getCodeDepths();
                bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                // ensure that the bitmaps size is not larger than the remaining file size
                bitmapsSize = min(static_cast<uint64_t>(bitmapsSize), fileSize - sizeof(DepthBitmapsHeader) - 1);
                inputFile.read(reinterpret_cast<char *>(_rawDepthBitmaps), bitmapsSize);
                
                alreadyReadBytes += sizeof(DepthBitmapsHeader);
                _compressedSizesExScan[0] = 0; // there is only one block
            }
            break;

        case MULTI_THREADED:
            {
                DepthBitmapsMultiThreadedHeader &header = reinterpret_cast<DepthBitmapsMultiThreadedHeader &>(_headerBuffer);
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsMultiThreadedHeader)); // read the rest of the header

                // get the threading info and compute the number of bytes used for transferring the compressed block sizes
                _numberOfCompressedBlocks = header.getNumberOfThreads();
                _bitsPerCompressedBlockSize = header.getBitsPerBlockSize();
                _numberOfBytesCompressedBlocks = (_numberOfCompressedBlocks * _bitsPerCompressedBlockSize + 7) / 8; // round to bytes
                inputFile.read(reinterpret_cast<char *>(_rawBlockSizes), _numberOfBytesCompressedBlocks); // read the compressed block sizes
                parseThreadingInfo(); // unpack the compressed block sizes

                // read the compressed depth bitmaps (the compressed Huffman tree)
                _usedDepths = header.getCodeDepths();
                bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                // ensure that the bitmaps size is not larger than the remaining file size
                bitmapsSize = min(static_cast<uint64_t>(bitmapsSize), fileSize - sizeof(DepthBitmapsMultiThreadedHeader) - _numberOfBytesCompressedBlocks - 1);
                inputFile.read(reinterpret_cast<char *>(_rawDepthBitmaps), bitmapsSize);

                alreadyReadBytes += sizeof(DepthBitmapsMultiThreadedHeader) + _numberOfBytesCompressedBlocks;
            }
            break;
        
        default: // corrupted header
            cerr << "Error: Unsupported header type" << endl;
            exit(1);
    }

    parseHuffmanTree(bitmapsSize); // decompress the Huffman tree from the compressed depth bitmaps
    alreadyReadBytes += bitmapsSize;
    fileSize -= alreadyReadBytes;
    inputFile.seekg(alreadyReadBytes, ios::beg); // skip the header, threading info (if present) and the compressed depth bitmaps

    BasicHeader &header = reinterpret_cast<BasicHeader &>(_headerBuffer);
    switch (header.getHeaderType() & ADAPTIVE)
    {
        case ADAPTIVE:
            // 2D size of the image
            _width = header.getWidth();
            _height = header.getHeight();
            _size = _width * _height;
            break;
        
        case STATIC:
            // 1D size of the image
            _size = header.getSize();
            break;

        default: // corrupted header
            cerr << "Error: Unsupported header type" << endl;
            exit(1);
    }

    _compressedData = reinterpret_cast<uint16_t *>(aligned_alloc(64, _size + 64));
    if (_compressedData == nullptr)
    {
        cerr << "Error: Could not allocate memory for compressed file" << endl;
        exit(1);
    }

    _decompressionBuffer = reinterpret_cast<uint8_t *>(aligned_alloc(64, _size + 64));
    if (_decompressionBuffer == nullptr)
    {
        cerr << "Error: Could not allocate memory for decompression" << endl;
        exit(1);
    }

    if (header.getHeaderType() & ADAPTIVE)
    {
        initializeBlockTypes(); // allocate memory for the block types
        uint32_t packedBlockCount = (_blockCount * BITS_PER_BLOCK_TYPE + 7) / 8; // round to bytes
        inputFile.read(reinterpret_cast<char *>(_decompressionBuffer), packedBlockCount); // read the packed block types
        fileSize -= packedBlockCount;

        for (uint32_t i = 0, j = 0; i < packedBlockCount; i++, j += 4) // unpack the block types
        {
            // 1 byte contains 4 block types (2 bits per block type)
            uint8_t blockTypes = _decompressionBuffer[i];
            _bestBlockTraversals[j] = static_cast<AdaptiveTraversals>(blockTypes >> 6);
            _bestBlockTraversals[j + 1] = static_cast<AdaptiveTraversals>((blockTypes >> 4) & 0b11);
            _bestBlockTraversals[j + 2] = static_cast<AdaptiveTraversals>((blockTypes >> 2) & 0b11);
            _bestBlockTraversals[j + 3] = static_cast<AdaptiveTraversals>(blockTypes & 0b11);
        }
    }

    inputFile.read(reinterpret_cast<char *>(_compressedData), fileSize); // read the remaining compressed data
    inputFile.close();
    
    return true;
}

void Decompressor::parseHuffmanTree(uint16_t &readBytes)
{
    DEBUG_PRINT("Parsing Huffman tree");

    uint8_t depthIdx = 0;
    uint16_t lastCode = -1;

    uint8_t missingDepth = _rawDepthBitmaps[0];
    uint16_t rawDepthBitmapsIdx = 1;
    uint8_t recoverDepthIdx = 0;

    for (uint8_t i = 1; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i)) // there are symbols at depth i
        {
            if (i == missingDepth) // this depth is the most populated and will we recovered later
            {
                recoverDepthIdx = depthIdx;
            }
            else
            {
                // recover the 32-bit mask indicating non-zero bytes in the 256-bit depth bitmap
                uint32_t mask = _rawDepthBitmaps[rawDepthBitmapsIdx++] << 24;
                mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++] << 16;
                mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++] << 8;
                mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++];

            // clear the current depth
            #if __AVX2__
                _symbolsAtDepths[depthIdx] = _mm256_setzero_si256();
                uint8_t *depthBytes = reinterpret_cast<uint8_t *>(_symbolsAtDepths + depthIdx);
            #else
                uint8_t *depthBytes = reinterpret_cast<uint8_t *>(_symbolsAtDepths + depthIdx);
                for (uint8_t j = 0; j < 32; j++)
                {
                    depthBytes[j] = 0;
                }
            #endif

                while (mask) // there are still non-zero bytes in the mask
                {
                    uint8_t firstSetBit = 31 - countl_zero(mask);
                    mask ^= 1U << firstSetBit; // mark the byte as processed
                    depthBytes[firstSetBit] = _rawDepthBitmaps[rawDepthBitmapsIdx++]; // read the byte
                }
            }
            depthIdx++;
        }
    }

    if (_usedDepths & 1) // symbols at depth 0, these were not seen in the compressed data, but they are necessary to recover the most populated depth
    {
        // same process of decompressing as above
        _usedDepths ^= 1;
        if (missingDepth != 0)
        {
            uint32_t mask = _rawDepthBitmaps[rawDepthBitmapsIdx++] << 24;
            mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++] << 16;
            mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++] << 8;
            mask |= _rawDepthBitmaps[rawDepthBitmapsIdx++];

            #if __AVX2__
                _symbolsAtDepths[depthIdx] = _mm256_setzero_si256();
                uint8_t *depthBytes = reinterpret_cast<uint8_t *>(_symbolsAtDepths + depthIdx);
            #else
                uint8_t *depthBytes = reinterpret_cast<uint8_t *>(_symbolsAtDepths + depthIdx);
                for (uint8_t j = 0; j < 32; j++)
                {
                    depthBytes[j] = 0;
                }
            #endif
            while (mask)
            {
                uint8_t firstSetBit = 31 - countl_zero(mask);
                mask ^= 1U << firstSetBit;
                depthBytes[firstSetBit] = _rawDepthBitmaps[rawDepthBitmapsIdx++];
            }
        }
    }
    readBytes = rawDepthBitmapsIdx;

    if (missingDepth != 0) // the missing depth is not the 0th depth (unseen symbols do not have to be recovered)
    {
        // fully populate the missing depth
    #if __AVX2__
        _symbolsAtDepths[recoverDepthIdx] = _mm256_set1_epi32(0xffff'ffff);
    #else
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + recoverDepthIdx);
        bits[0] = 0xffff'ffff'ffff'ffff;
        bits[1] = 0xffff'ffff'ffff'ffff;
        bits[2] = 0xffff'ffff'ffff'ffff;
        bits[3] = 0xffff'ffff'ffff'ffff;
    #endif

        // XOR the other depths with the missing depth to recover the set bits
        for (uint8_t i = 0; i <= depthIdx; i++)
        {
            if (i != recoverDepthIdx) // do not XOR the missing depth with itself
            {
            #if __AVX2__
                _symbolsAtDepths[recoverDepthIdx] = _mm256_xor_si256(_symbolsAtDepths[recoverDepthIdx], _symbolsAtDepths[i]);
            #else
                uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + recoverDepthIdx);
                uint64_t *otherBits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
                bits[0] ^= otherBits[0];
                bits[1] ^= otherBits[1];
                bits[2] ^= otherBits[2];
                bits[3] ^= otherBits[3];
            #endif
            }
        }
    }

    // recover the Huffman tree and create lookup tables for decompression
#if __AVX512BW__ && __AVX512VL__
    uint8_t lastDepth = 0;
    uint8_t masksIdx = 0;
    depthIdx = 0;
    for (uint8_t i = 0; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i)) // there are symbols at depth i
        {
            // get the number of symbols at the current depth
        #if __AVX512VPOPCNTDQ__
            uint64v4_t popCounts = _mm256_popcnt_epi64(_symbolsAtDepths[depthIdx]);
            uint64_t *bits = reinterpret_cast<uint64_t *>(&popCounts);
            uint16_t sum1 = bits[0] + bits[1];
            uint16_t sum2 = bits[2] + bits[3];
            uint16_t symbolCount = sum1 + sum2;
        #else
            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + depthIdx);
            uint32_t symbolsToProcess = _mm_popcnt_u64(bits[0]) + _mm_popcnt_u64(bits[1]) + _mm_popcnt_u64(bits[2]) + _mm_popcnt_u64(bits[3]);
        #endif

            lastCode = (lastCode + 1) << (i - lastDepth); // compute the canonical Huffman code for the 1st symbol at the current depth
            lastDepth = i; // update last seen depth

            while (symbolsToProcess) // remaining symbols at the current depth, more than 1 prefix can appear at the same depth
            {
                uint8_t trailingZeros = countr_zero(lastCode); // 2^trailingZeros is the number of symbols that can be encoded with the current prefix
                uint32_t fits = min(1U << trailingZeros, symbolsToProcess); // compute how many symbols will be encoded with the current prefix

                // compute the prefix length
                uint8_t leadingZeros = 64 - countl_zero(fits);
                uint8_t minZeros = min(min(leadingZeros, i), trailingZeros); // solve the case when lastCode is 0
                uint16_t prefixLength = i - minZeros;

                lastCode += fits;
                symbolsToProcess -= fits;

                // store information about this prefix at the current depth for later use
                _depthsIndices[masksIdx].depth = i;
                _depthsIndices[masksIdx].prefixLength = prefixLength;
                _depthsIndices[masksIdx].symbolsAtDepthIndex = depthIdx;
                // create a dictionary between prefix length and index in the depthsIndices array
                _prefixIndexLengths[masksIdx].index = masksIdx;
                _prefixIndexLengths[masksIdx].length = _depthsIndices[masksIdx].prefixLength;

                masksIdx++;
            }

            lastCode--;
            depthIdx++;
        }
    }
    // sort the prefixes by their length in descending order (longest prefix first)
    sort(_prefixIndexLengths, _prefixIndexLengths + masksIdx, 
            [](const PrefixIndexLength &a, const PrefixIndexLength &b) { return a.length > b.length; });
    for (uint8_t i = 0; i < masksIdx; i++)
    {
        // ensure that prefixes and masks at different depths are stored in descending order to the final vectors
        _depthsIndices[_prefixIndexLengths[i].index].masksIndex = i;
    }
    
    lastDepth = 0;
    lastCode = -1;
    uint16_t symbolIdx = 0;
    // crate the vectors of prefixes and masks for the AVX instructions
    for (uint8_t i = 0; i < masksIdx; i++)
    {
        uint8_t delta = _depthsIndices[i].depth - lastDepth;
        lastDepth = _depthsIndices[i].depth;
        lastCode = (lastCode + 1) << delta;
        // the prefix is given by the first canonical Huffman code
        _codePrefixes[31 - _depthsIndices[i].masksIndex] = lastCode << (16 - _depthsIndices[i].depth);
        // the mask is given by the length of the prefix
        _codeMasks[31 - _depthsIndices[i].masksIndex] = (~0U) << (16 - _depthsIndices[i].prefixLength);

        // fill in lookup table for the given prefix
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].prefixIndex = symbolIdx;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].suffixShift = 16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].prefixShift = _depthsIndices[i].prefixLength;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].codeLength = _depthsIndices[i].depth;

        int32_t maxSymbols = 1 << countr_zero(lastCode); // compute the maximum number of symbols that can be encoded with the current prefix
        uint16_t lastSymbolIdx = symbolIdx;
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + _depthsIndices[i].symbolsAtDepthIndex);
        for (uint8_t j = 0; j < 4; j++)
        {
            uint8_t symbol = j << 6; // adjust offset of the symbols, depending on the position in the 256-bit bitmap
            uint8_t leadingZeros;
            while (maxSymbols > 0 && (leadingZeros = countl_zero(bits[j])) < 64) // there is space in the prefix and there are symbols left to encode
            {
                maxSymbols--;
                uint8_t adjustedSymbol = symbol + leadingZeros;
                bits[j] ^= 1UL << (63 - leadingZeros); // mark the symbol as processed
                _symbolsTable[symbolIdx++] = adjustedSymbol; // store the symbol in the symbols lookup table
            }
        }
        lastCode += symbolIdx - lastSymbolIdx - 1; // update the last code by the number of symbols encoded with the current prefix
    }
#else
    // the compressor balanced the tree in a way that the prefixes always start with a sequence of 1 bits followed by a 0 bit
    lastCode = -1;
    depthIdx = 0;
    uint16_t symbolIdx = 0;
    uint8_t delta = 0;
    for (uint8_t i = 0; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i))
        {
            lastCode = (lastCode + 1) << delta;
            uint16_t lastSymbolIdx = symbolIdx;

            // store information about the current prefix in the lookup table
            _indexShiftsCodeLengths[depthIdx].prefixIndex = symbolIdx;
            _indexShiftsCodeLengths[depthIdx].suffixShift = 16 - i + depthIdx + 1;
            _indexShiftsCodeLengths[depthIdx].prefixShift = depthIdx + 1;
            _indexShiftsCodeLengths[depthIdx].codeLength = i;

            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + depthIdx);
            for (uint8_t j = 0; j < 4; j++)
            {
                uint8_t symbol = j << 6; // adjust offset of the symbols, depending on the position in the 256-bit bitmap
                uint8_t leadingZeros;
                while ((leadingZeros = countl_zero(bits[j])) < 64)
                {
                    uint8_t adjustedSymbol = symbol + leadingZeros;
                    bits[j] ^= 1UL << (63 - leadingZeros); // mark the symbol as processed
                    _symbolsTable[symbolIdx++] = adjustedSymbol; // store the symbol in the symbols lookup table
                }
            }

            lastCode += symbolIdx - lastSymbolIdx - 1; // update the last code by the number of symbols encoded with the current prefix
            delta = 0;
            depthIdx++;
        }
        delta++;
    }
#endif
    DEBUG_PRINT("Huffman tree parsed");
}

void Decompressor::parseThreadingInfo()
{
    DEBUG_PRINT("Parsing threading info");
    uint8_t packedIdx = 0;
    uint8_t chunkIdx = 0;

    for (uint32_t i = 0; i < _numberOfCompressedBlocks; i++)
    {
        _compressedSizes[i] = 0;
        uint8_t bits = _bitsPerCompressedBlockSize;

        do
        {
            uint8_t readBits = bits < 8 - chunkIdx ? bits : 8 - chunkIdx;
            _compressedSizes[i] <<= readBits; // move the current content towards the MSB
            _compressedSizes[i] |= _rawBlockSizes[packedIdx] >> (8 - readBits); // place the extracted bits to the lower bits
            _rawBlockSizes[packedIdx] <<= readBits; // shift out the read bits
            bits -= readBits;
            chunkIdx += readBits;
            if (chunkIdx >= 8) // byte was fully read
            {
                chunkIdx &= 0x07;
                packedIdx++; // move to the next byte
            }
        }
        while (bits); // there more bits to unpack
    }

    // exclusive scan over the compressed sizes
    _compressedSizesExScan[0] = 0;
    for (uint32_t i = 0; i < _numberOfCompressedBlocks; i++)
    {
        _compressedSizesExScan[i + 1] = _compressedSizesExScan[i] + _compressedSizes[i];
    }
    DEBUG_PRINT("Threading info parsed");
}

void Decompressor::transformRLE(uint16_t *compressedData, symbol_t *decompressedData, uint64_t bytesToDecompress)
{
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " transforming RLE");
    // notice in both implementations that the number of if statements is minimized and decisions are hidden in computations

#if __AVX512BW__ && __AVX512VL__ // the AVX prefix matching decompression
    // declare state variables
    uint64_t decompressedIdx = 0;
    bool overflow; // flag indicating that the current bits overflow to the next short, indices must be incremented
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint8_t sameSymbolCount = 0;

    // retrieve the vectors of prefixes and masks assembled above
    uint16v32_t prefixes = _codePrefixesVector;
    uint16v32_t masks = _codeMasksVector;

    // process the first symbol, avoid using if statements
    uint16_t current = compressedData[currentIdx];
    uint16v32_t currentVector = _mm512_set1_epi16(current);      // broadcast the current symbol to an AVX vector
    uint16v32_t masked = _mm512_and_si512(currentVector, masks); // mask the current bits to the lengths of each prefix
    uint16v32_t xored = _mm512_xor_si512(prefixes, masked);      // XOR the masked current bits
    uint32_t prefixBitMap = _mm512_cmp_epi16_mask(xored, _mm512_setzero_si512(), _MM_CMPINT_EQ); // get a bitmap of matching prefixes
    uint8_t prefixIdx = countl_zero(prefixBitMap); // find the index of the first matching prefix
    IndexShiftsCodeLength codeMatch = _indexShiftsCodeLengths[prefixIdx]; // search for the matching prefix in the lookup table
    uint8_t suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift; // extract the suffix from the current bits
    symbol_t symbol = _symbolsTable[codeMatch.prefixIndex + suffix]; // get the symbol from the symbols lookup table
    decompressedData[decompressedIdx++] = symbol; // store the symbol in the decompressed data
    bitLength += codeMatch.codeLength; // update the bit length by the length of the extracted code
    symbol_t lastSymbol = symbol;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < bytesToDecompress) // there are still data to decompress
    {
        // concatenate the current bits from lower bits of the current short and upper bits of the next short
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);

        // rest of the process is the same as above 
        // FIXME: refactor the code to avoid code duplication
        currentVector = _mm512_set1_epi16(current);
        masked = _mm512_and_si512(currentVector, masks);
        xored = _mm512_xor_si512(prefixes, masked);
        prefixBitMap = _mm512_cmp_epi16_mask(xored, _mm512_setzero_si512(), _MM_CMPINT_EQ);
        prefixIdx = countl_zero(prefixBitMap);
        codeMatch = _indexShiftsCodeLengths[prefixIdx];
        suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
        symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
        decompressedData[decompressedIdx++] = symbol;

        // update same symbol count, make the value available as soon as possible so the jump condition can be evaluated and jump prediction is not necessary
        sameSymbolCount += lastSymbol == symbol;
        sameSymbolCount >>= lastSymbol != symbol; // reset the count if the symbols differ (avoid conditional jumps)
        lastSymbol = symbol;

        bitLength += codeMatch.codeLength;
        overflow = bitLength >= 16;
        bitLength &= 0x0f; // modulo 16
        // increment the indices if the current bits overflow to the next short (avoid conditional jumps)
        nextIdx += overflow;    
        currentIdx += overflow;
        inverseBitLength = 16 - bitLength;

        if (sameSymbolCount == 2) // 3 same symbols in a row
        {
            sameSymbolCount = 0;
            bool repeat;
            uint8_t multiplier = 0;
            uint64_t totalRepetitions = 0;
            do
            {
                current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
                repeat = current & 0x8000; // set MSB indicates if the repetition continues
                uint64_t repetitions = ((current & 0x7FFF) >> (16 - BITS_PER_REPETITION_NUMBER)); // align the repetitions to LSB
                repetitions <<= multiplier;
                totalRepetitions += repetitions;

                bitLength += BITS_PER_REPETITION_NUMBER;
                overflow = bitLength >= 16;
                bitLength &= 0x0f; // modulo 16
                nextIdx += overflow;
                currentIdx += overflow;
                inverseBitLength = 16 - bitLength;
                // in each iteration the number of repetitions is multiplied by 2^(BITS_PER_REPETITION_NUMBER - 1)
                multiplier += (BITS_PER_REPETITION_NUMBER - 1);
            }
            while (repeat); // still more repetitions of the same symbol

            #pragma omp simd aligned(decompressedData: 64) simdlen(64)
            for (uint64_t i = 0; i < totalRepetitions; i++) // store the repeated symbol in the decompressed data
            {
                decompressedData[decompressedIdx++] = symbol;
            }
        }
    }
#else // the backup decompression, which count leading ones in prefixes
    // declare state variables
    uint64_t decompressedIdx = 0;
    bool overflow;
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint8_t sameSymbolCount = 0;

    uint16_t current = compressedData[currentIdx];
    uint8_t prefixIdx = countl_one(current); // count the leading ones in the current bits to get the prefix index
    IndexShiftsCodeLength codeMatch = _indexShiftsCodeLengths[prefixIdx]; // search for the matching prefix in the lookup table
    uint8_t suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift; // extract the suffix from the current bits
    symbol_t symbol = _symbolsTable[codeMatch.prefixIndex + suffix]; // get the symbol from the symbols lookup table
    decompressedData[decompressedIdx++] = symbol; // store the symbol in the decompressed data
    bitLength += codeMatch.codeLength; // update the bit length by the length of the extracted code
    symbol_t lastSymbol = symbol;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < bytesToDecompress)
    {
        // concatenate the current bits from lower bits of the current short and upper bits of the next short
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);

        // rest of the process is the same as above 
        // FIXME: refactor the code to avoid code duplication
        prefixIdx = countl_one(current);
        codeMatch = _indexShiftsCodeLengths[prefixIdx];
        suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
        symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
        decompressedData[decompressedIdx++] = symbol;

        // update same symbol count, make the value available as soon as possible so the jump condition can be evaluated and jump prediction is not necessary
        sameSymbolCount += lastSymbol == symbol;
        sameSymbolCount >>= lastSymbol != symbol; // reset the count if the symbols differ (avoid conditional jumps)
        lastSymbol = symbol;

        bitLength += codeMatch.codeLength;
        overflow = bitLength >= 16;
        bitLength &= 0x0f; // modulo 16
        // increment the indices if the current bits overflow to the next short (avoid conditional jumps)
        nextIdx += overflow;    
        currentIdx += overflow;
        inverseBitLength = 16 - bitLength;

        if (sameSymbolCount == 2) // 3 same symbols in a row
        {
            sameSymbolCount = 0;
            bool repeat;
            uint8_t multiplier = 0;
            uint64_t totalRepetitions = 0;
            do
            {
                current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
                repeat = current & 0x8000; // set MSB indicates if the repetition continues
                uint64_t repetitions = ((current & 0x7FFF) >> (16 - BITS_PER_REPETITION_NUMBER)); // align the repetitions to LSB
                repetitions <<= multiplier;
                totalRepetitions += repetitions;

                bitLength += BITS_PER_REPETITION_NUMBER;
                overflow = bitLength >= 16;
                bitLength &= 0x0f; // modulo 16
                nextIdx += overflow;
                currentIdx += overflow;
                inverseBitLength = 16 - bitLength;
                // in each iteration the number of repetitions is multiplied by 2^(BITS_PER_REPETITION_NUMBER - 1)
                multiplier += (BITS_PER_REPETITION_NUMBER - 1);
            }
            while (repeat); // still more repetitions of the same symbol

            #pragma omp simd aligned(decompressedData: 64) simdlen(64)
            for (uint64_t i = 0; i < totalRepetitions; i++) // store the repeated symbol in the decompressed data
            {
                decompressedData[decompressedIdx++] = symbol;
            }
        }
    }
#endif
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " RLE transformed, " << decompressedIdx << " bytes decompressed");
}

void Decompressor::reverseDifferenceModel(symbol_t *source, symbol_t *destination, uint64_t bytesToProcess)
{
    destination[0] = source[0];
    #pragma omp simd aligned(source, destination: 64) simdlen(64)
    for (uint64_t i = 1; i < bytesToProcess; i++)
    {
        destination[i] = source[i] + destination[i - 1]; // get the current symbol by adding the previous symbol to the decompressed current symbol
    }
}

void Decompressor::decompressStaticModel()
{
    decompressStatic(); // use the regular static decompression

    #pragma omp master
    {
        DEBUG_PRINT("Static decompression with model");
        // tasks are necessary, there might be less threads available for decompression than were used for compression
        
        // distribute the work
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks; // round up
        bytesPerBlock += bytesPerBlock & 0b1; // ensure that the number of bytes is even
        uint64_t bytesPerLastBlock = _size - bytesPerBlock * (_numberOfCompressedBlocks - 1); 

        symbol_t *source = _decompressedData;
        symbol_t *destination = reinterpret_cast<symbol_t *>(_compressedData);

        // schedule the model reversal of each block as a task
        for (uint8_t i = 0; i < _numberOfCompressedBlocks - 1; i++)
        {
            #pragma omp task firstprivate(i)
            {
                reverseDifferenceModel(source + i * bytesPerBlock, destination + i * bytesPerBlock, bytesPerBlock);
            }
        }

        #pragma omp task // last block may be smaller
        {
            reverseDifferenceModel(source + (_numberOfCompressedBlocks - 1) * bytesPerBlock, 
                                   destination + (_numberOfCompressedBlocks - 1) * bytesPerBlock, bytesPerLastBlock);
        }

        _decompressedData = destination;
    }
    #pragma omp taskwait
    #pragma omp barrier
}

void Decompressor::decompressAdaptiveModel()
{
    decompressAdaptive(); // use the regular adaptive decompression

    #pragma omp master
    {
        DEBUG_PRINT("Adaptive decompression with model");
        // tasks are necessary, there might be less threads available for decompression than were used for compression

        // distribute the work
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
        bytesPerBlock += bytesPerBlock & 0b1; // ensure that the number of bytes is even
        uint64_t bytesPerLastBlock = _size - bytesPerBlock * (_numberOfCompressedBlocks - 1);

        symbol_t *source = _decompressedData;
        symbol_t *destination = _decompressionBuffer;
        // schedule the decompression of each block as a task
        for (uint8_t i = 0; i < _numberOfCompressedBlocks - 1; i++)
        {
            #pragma omp task firstprivate(i)
            {
                reverseDifferenceModel(source + i * bytesPerBlock, destination + i * bytesPerBlock, bytesPerBlock);
            }
        }

        #pragma omp task // last block may be smaller
        {
            reverseDifferenceModel(source + (_numberOfCompressedBlocks - 1) * bytesPerBlock, destination + (_numberOfCompressedBlocks - 1) * bytesPerBlock, 
                                   bytesPerLastBlock);
        }

        _decompressedData = destination;
    }
    #pragma omp taskwait
    #pragma omp barrier
}

void Decompressor::decompressStatic()
{
    #pragma omp master
    {
        DEBUG_PRINT("Static decompression");
        // tasks are necessary, there might be less threads available for decompression than were used for compression

        // distribute the work
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
        bytesPerBlock += bytesPerBlock & 0b1; // ensure that the number of bytes is even
        uint64_t bytesPerLastBlock = _size - bytesPerBlock * (_numberOfCompressedBlocks - 1);

        // schedule the decompression of each block as a task
        for (uint8_t i = 0; i < _numberOfCompressedBlocks - 1; i++)
        {
            #pragma omp task firstprivate(i)
            {
                transformRLE(_compressedData + (_compressedSizesExScan[i] >> 1), _decompressionBuffer + i * bytesPerBlock, bytesPerBlock);
            }
        }

        #pragma omp task // last block may be smaller
        {
            transformRLE(_compressedData + (_compressedSizesExScan[_numberOfCompressedBlocks - 1] >> 1), 
                         _decompressionBuffer + (_numberOfCompressedBlocks - 1) * bytesPerBlock, bytesPerLastBlock);
        }

        _decompressedData = _decompressionBuffer;
    }
    #pragma omp taskwait
    #pragma omp barrier
}

void Decompressor::decompressAdaptive()
{
    decompressStatic();

    #pragma omp master
    {
        DEBUG_PRINT("Adaptive decompression");

        symbol_t *source = _decompressionBuffer;
        symbol_t *destination = reinterpret_cast<symbol_t *>(_compressedData);

        for (uint32_t i = 0; i < _blocksPerColumn; i++)
        {
            for (uint32_t j = 0; j < _blocksPerRow; j++)
            {
                // schedule the decompression of each block as a task
                switch (_bestBlockTraversals[i * _blocksPerRow + j])
                {
                case HORIZONTAL:
                    #pragma omp task firstprivate(i, j)
                    {
                        deserializeBlock(source, destination, i, j);
                    }
                    break;
                
                case VERTICAL:
                    #pragma omp task firstprivate(i, j)
                    {
                        transposeDeserializeBlock(source, destination, i, j);
                    }
                    break;

                default:
                    cerr << "Error: Unsupported traversal type" << endl;
                    exit(1);
                    break;
                }
            }
        }
        _decompressedData = destination;
        DEBUG_PRINT("Adaptive decompression done");
    }
    #pragma omp taskwait
    #pragma omp barrier
}

void Decompressor::writeOutputFile(std::string outputFileName)
{
    ofstream outputFile(outputFileName, ios::binary);
    if (!outputFile.is_open())
    {
        cout << "Error: Could not open output file " << outputFileName << endl;
        exit(1);
    }

    outputFile.write(reinterpret_cast<char *>(_decompressedData), _size);
    outputFile.close();
}

void Decompressor::decompress(string inputFileName, string outputFileName)
{
    if (readInputFile(inputFileName, outputFileName)) // the data are compressed 
    {
        #pragma omp parallel // launch threads, i.e. create a parallel region
        {
            // use the correct decompression method based on the header type
            if (_header.getHeaderType() & MODEL)
            {
                if (_header.getHeaderType() & ADAPTIVE)
                {
                    decompressAdaptiveModel();
                }
                else
                {
                    decompressStaticModel();
                }
            }
            else
            {
                if (_header.getHeaderType() & ADAPTIVE)
                {
                    decompressAdaptive();
                }
                else
                {
                    decompressStatic();
                }
            }
        }
        
        writeOutputFile(outputFileName);
    }
}
