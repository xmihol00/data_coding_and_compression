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

        char *buffer = new char[fileSize];
        inputFile.read(buffer, fileSize - 1);
        outputFile.write(buffer, fileSize - 1);

        delete[] buffer;
        outputFile.close();
        inputFile.close();
        
        return false;
    }

    inputFile.seekg(0, ios::beg);
    uint16_t bitmapsSize;
    switch (firstByte.getHeaderType() & MULTI_THREADED)
    {
        case SINGLE_THREADED:
            {
                DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_headerBuffer);
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsHeader));
                _usedDepths = header.getCodeDepths();
                bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                inputFile.read(reinterpret_cast<char *>(_rawDepthBitmaps), bitmapsSize);
                alreadyReadBytes += sizeof(DepthBitmapsHeader);
                _compressedSizesExScan[0] = 0;
            }
            break;

        case MULTI_THREADED:
            {
                DepthBitmapsMultiThreadedHeader &header = reinterpret_cast<DepthBitmapsMultiThreadedHeader &>(_headerBuffer);
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsMultiThreadedHeader));
                _numberOfCompressedBlocks = header.getNumberOfThreads();
                _bitsPerCompressedBlockSize = header.getBitsPerBlockSize();
                _numberOfBytesCompressedBlocks = (_numberOfCompressedBlocks * _bitsPerCompressedBlockSize + 7) / 8;
                inputFile.read(reinterpret_cast<char *>(_rawBlockSizes), _numberOfBytesCompressedBlocks);
                _usedDepths = header.getCodeDepths();
                bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                inputFile.read(reinterpret_cast<char *>(_rawDepthBitmaps), bitmapsSize);
                alreadyReadBytes += sizeof(DepthBitmapsMultiThreadedHeader) + _numberOfBytesCompressedBlocks;
                parseThreadingInfo();
            }
            break;
        
        default:
            cerr << "Error: Unsupported header type" << endl;
            exit(1);
    }

    parseHuffmanTree(bitmapsSize);
    alreadyReadBytes += bitmapsSize;
    fileSize -= alreadyReadBytes;
    inputFile.seekg(alreadyReadBytes, ios::beg);

    BasicHeader &header = reinterpret_cast<BasicHeader &>(_headerBuffer);
    switch (header.getHeaderType() & ADAPTIVE)
    {
        case ADAPTIVE:
            _width = header.getWidth();
            _height = header.getHeight();
            _size = _width * _height;
            break;
        
        case STATIC:
            _size = header.getSize();
            break;

        default:
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
        initializeBlockTypes();
        uint32_t packedBlockCount = (_blockCount * BITS_PER_BLOCK_TYPE + 7) / 8;
        inputFile.read(reinterpret_cast<char *>(_decompressionBuffer), packedBlockCount);
        fileSize -= packedBlockCount;

        for (uint32_t i = 0, j = 0; i < packedBlockCount; i++, j += 4)
        {
            uint8_t blockTypes = _decompressionBuffer[i];
            _bestBlockTraversals[j] = static_cast<AdaptiveTraversals>(blockTypes >> 6);
            _bestBlockTraversals[j + 1] = static_cast<AdaptiveTraversals>((blockTypes >> 4) & 0b11);
            _bestBlockTraversals[j + 2] = static_cast<AdaptiveTraversals>((blockTypes >> 2) & 0b11);
            _bestBlockTraversals[j + 3] = static_cast<AdaptiveTraversals>(blockTypes & 0b11);
        }
    }

    inputFile.read(reinterpret_cast<char *>(_compressedData), fileSize);
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
        if (_usedDepths & (1UL << i))
        {
            if (i == missingDepth)
            {
                recoverDepthIdx = depthIdx;
            }
            else
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
            depthIdx++;
        }
    }

    if (_usedDepths & 1)
    {
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

    if (missingDepth != 0)
    {
    #if __AVX2__
        _symbolsAtDepths[recoverDepthIdx] = _mm256_set1_epi32(0xffff'ffff);
    #else
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + recoverDepthIdx);
        bits[0] = 0xffff'ffff'ffff'ffff;
        bits[1] = 0xffff'ffff'ffff'ffff;
        bits[2] = 0xffff'ffff'ffff'ffff;
        bits[3] = 0xffff'ffff'ffff'ffff;
    #endif
        for (uint8_t i = 0; i <= depthIdx; i++)
        {
            if (i != recoverDepthIdx)
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

#if __AVX512BW__ && __AVX512VL__
    uint8_t lastDepth = 0;
    uint8_t masksIdx = 0;
    depthIdx = 0;
    for (uint8_t i = 0; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i))
        {
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
            lastCode = (lastCode + 1) << (i - lastDepth);
            lastDepth = i;

            while (symbolsToProcess)
            {
                uint8_t trailingZeros = countr_zero(lastCode);
                uint32_t fits = min(1U << trailingZeros, symbolsToProcess);
                uint8_t leadingZeros = 64 - countl_zero(fits);
                uint8_t minZeros = min(min(leadingZeros, i), trailingZeros);
                uint16_t prefixLength = i - minZeros;
                lastCode += fits;
                symbolsToProcess -= fits;

                _depthsIndices[masksIdx].depth = i;
                _depthsIndices[masksIdx].prefixLength = prefixLength;
                _depthsIndices[masksIdx].symbolsAtDepthIndex = depthIdx;
                _depthsIndices[masksIdx].masksIndex = masksIdx;
                _indexPrefixLengths[masksIdx].index = masksIdx;
                _indexPrefixLengths[masksIdx].prefixLength = _depthsIndices[masksIdx].prefixLength;

                masksIdx++;
            }

            lastCode--;
            depthIdx++;
        }
    }
    sort(_indexPrefixLengths, _indexPrefixLengths + masksIdx, 
            [](const IndexPrefixLength &a, const IndexPrefixLength &b) { return a.prefixLength > b.prefixLength; });
    for (uint8_t i = 0; i < masksIdx; i++)
    {
        _depthsIndices[_indexPrefixLengths[i].index].masksIndex = i;
    }
    
    lastDepth = 0;
    lastCode = -1;
    uint16_t symbolIdx = 0;
    for (uint8_t i = 0; i < masksIdx; i++)
    {
        uint8_t delta = _depthsIndices[i].depth - lastDepth;
        lastDepth = _depthsIndices[i].depth;
        lastCode = (lastCode + 1) << delta;
        _codePrefixes[31 - _depthsIndices[i].masksIndex] = lastCode << (16 - _depthsIndices[i].depth);
        _codeMasks[31 - _depthsIndices[i].masksIndex] = (~0U) << (16 - _depthsIndices[i].prefixLength);
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].prefixIndex = symbolIdx;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].suffixShift = 16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].prefixShift = _depthsIndices[i].prefixLength;
        _indexShiftsCodeLengths[_depthsIndices[i].masksIndex].codeLength = _depthsIndices[i].depth;

        int32_t maxSymbols = 1 << countr_zero(lastCode); 
        uint16_t lastSymbolIdx = symbolIdx;
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + _depthsIndices[i].symbolsAtDepthIndex);
        for (uint8_t j = 0; j < 4; j++)
        {
            uint8_t symbol = j << 6;
            uint8_t leadingZeros;
            while (maxSymbols > 0 && (leadingZeros = countl_zero(bits[j])) < 64)
            {
                maxSymbols--;
                uint8_t adjustedSymbol = symbol + leadingZeros;
                bits[j] ^= 1UL << (63 - leadingZeros);
                _symbolsTable[symbolIdx++] = adjustedSymbol;
            }
        }
        lastCode += symbolIdx - lastSymbolIdx - 1;
    }
#else
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
            _indexShiftsCodeLengths[depthIdx].prefixIndex = symbolIdx;
            _indexShiftsCodeLengths[depthIdx].suffixShift = 16 - i + depthIdx + 1;
            _indexShiftsCodeLengths[depthIdx].prefixShift = depthIdx + 1;
            _indexShiftsCodeLengths[depthIdx].codeLength = i;

            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + depthIdx);
            for (uint8_t j = 0; j < 4; j++)
            {
                uint8_t symbol = j << 6;
                uint8_t leadingZeros;
                while ((leadingZeros = countl_zero(bits[j])) < 64)
                {
                    uint8_t adjustedSymbol = symbol + leadingZeros;
                    bits[j] ^= 1UL << (63 - leadingZeros);
                    _symbolsTable[symbolIdx++] = adjustedSymbol;
                }
            }

            lastCode += symbolIdx - lastSymbolIdx - 1;
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
            _compressedSizes[i] <<= readBits;
            _compressedSizes[i] |= _rawBlockSizes[packedIdx] >> (8 - readBits);
            _rawBlockSizes[packedIdx] <<= readBits;
            bits -= readBits;
            chunkIdx += readBits;
            if (chunkIdx >= 8)
            {
                chunkIdx &= 0x07;
                packedIdx++;
            }
        }
        while (bits);
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
    uint64_t decompressedIdx = 0;
#if __AVX512BW__ && __AVX512VL__
    bool overflow;
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint16v32_t prefixes = _codePrefixesVector;
    uint16v32_t masks = _codeMasksVector;
    uint16_t current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
    DEBUG_PRINT("Current: " << bitset<16>(current));
    uint16v32_t currentVector = _mm512_set1_epi16(current);
    uint16v32_t masked = _mm512_and_si512(currentVector, masks);
    uint16v32_t xored = _mm512_xor_si512(prefixes, masked);
    uint32_t prefixBitMap = _mm512_cmp_epi16_mask(xored, _mm512_setzero_si512(), _MM_CMPINT_EQ);
    DEBUG_PRINT("PrefixBitMap: " << bitset<16>(prefixBitMap));
    uint8_t prefixIdx = countl_zero(prefixBitMap);
    IndexShiftsCodeLength codeMatch = _indexShiftsCodeLengths[prefixIdx];
    DEBUG_PRINT("PrefixIdx: " << (int)prefixIdx);
    DEBUG_PRINT("PrefixLength: " << (int)codeMatch.prefixShift);
    DEBUG_PRINT("SuffixShift: " << (int)codeMatch.suffixShift);
    uint8_t suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
    symbol_t symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
    decompressedData[decompressedIdx++] = symbol;
    DEBUG_PRINT("Symbol: " << (char)symbol << " " << (int)symbol << "\n");
    bitLength += codeMatch.codeLength;
    symbol_t lastSymbol = symbol;
    uint8_t sameSymbolCount = 0;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < bytesToDecompress)
    {
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
        currentVector = _mm512_set1_epi16(current);
        masked = _mm512_and_si512(currentVector, masks);
        xored = _mm512_xor_si512(prefixes, masked);
        prefixBitMap = _mm512_cmp_epi16_mask(xored, _mm512_setzero_si512(), _MM_CMPINT_EQ);
        prefixIdx = countl_zero(prefixBitMap);
        codeMatch = _indexShiftsCodeLengths[prefixIdx];
        suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
        symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
        decompressedData[decompressedIdx++] = symbol;
        sameSymbolCount += lastSymbol == symbol;
        sameSymbolCount >>= lastSymbol != symbol;
        lastSymbol = symbol;

        bitLength += codeMatch.codeLength;
        overflow = bitLength >= 16;
        bitLength &= 0x0f;
        nextIdx += overflow;
        currentIdx += overflow;
        inverseBitLength = 16 - bitLength;

        if (sameSymbolCount == 2)
        {
            sameSymbolCount = 0;
            bool repeat;
            uint8_t multiplier = 0;
            do
            {
                current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
                repeat = current & 0x8000;
                uint32_t repetitions = ((current & 0x7FFF) >> (16 - BITS_PER_REPETITION_NUMBER));
                repetitions <<= multiplier;

                for (uint32_t i = 0; i < repetitions; i++)
                {
                    decompressedData[decompressedIdx++] = symbol;
                }

                bitLength += BITS_PER_REPETITION_NUMBER;
                overflow = bitLength >= 16;
                bitLength &= 0x0F;
                nextIdx += overflow;
                currentIdx += overflow;
                inverseBitLength = 16 - bitLength;
                multiplier += (BITS_PER_REPETITION_NUMBER - 1);
            }
            while (repeat);
        }
    }
#else
    bool overflow;
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint16_t current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
    uint8_t prefixIdx = countl_one(current);
    IndexShiftsCodeLength codeMatch = _indexShiftsCodeLengths[prefixIdx];
    uint8_t suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
    symbol_t symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
    decompressedData[decompressedIdx++] = symbol;
    bitLength += codeMatch.codeLength;
    symbol_t lastSymbol = symbol;
    uint8_t sameSymbolCount = 0;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < bytesToDecompress)
    {
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
        prefixIdx = countl_one(current);
        codeMatch = _indexShiftsCodeLengths[prefixIdx];
        suffix = static_cast<uint16_t>(current << codeMatch.prefixShift) >> codeMatch.suffixShift;
        symbol = _symbolsTable[codeMatch.prefixIndex + suffix];
        decompressedData[decompressedIdx++] = symbol;
        sameSymbolCount += lastSymbol == symbol;
        sameSymbolCount >>= lastSymbol != symbol;
        lastSymbol = symbol;

        bitLength += codeMatch.codeLength;
        overflow = bitLength >= 16;
        bitLength &= 0x0f;
        nextIdx += overflow;
        currentIdx += overflow;
        inverseBitLength = 16 - bitLength;

        if (sameSymbolCount == 2)
        {
            sameSymbolCount = 0;
            bool repeat;
            uint8_t multiplier = 0;
            do
            {
                current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
                repeat = current & 0x8000;
                uint32_t repetitions = ((current & 0x7FFF) >> (16 - BITS_PER_REPETITION_NUMBER));
                repetitions <<= multiplier;

                for (uint32_t i = 0; i < repetitions; i++)
                {
                    decompressedData[decompressedIdx++] = symbol;
                }

                bitLength += BITS_PER_REPETITION_NUMBER;
                overflow = bitLength >= 16;
                bitLength &= 0x0F;
                nextIdx += overflow;
                currentIdx += overflow;
                inverseBitLength = 16 - bitLength;
                multiplier += (BITS_PER_REPETITION_NUMBER - 1);
            }
            while (repeat);
        }
    }
#endif
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " RLE transformed");
}

void Decompressor::reverseDifferenceModel(symbol_t *source, symbol_t *destination, uint64_t bytesToProcess)
{
    destination[0] = source[0];
    #pragma omp simd aligned(source, destination: 64) simdlen(64)
    for (uint32_t i = 1; i < bytesToProcess; i++)
    {
        destination[i] = source[i] + destination[i - 1];
    }
}

void Decompressor::decompressStaticModel()
{
    decompressStatic();

    #pragma omp master
    {
        DEBUG_PRINT("Static decompression with model");
        
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
        uint64_t bytesPerLastBlock = _size - bytesPerBlock * (_numberOfCompressedBlocks - 1);

        symbol_t *source = _decompressedData;
        symbol_t *destination = reinterpret_cast<symbol_t *>(_compressedData);

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
            reverseDifferenceModel(source + (_numberOfCompressedBlocks - 1) * bytesPerBlock, destination + (_numberOfCompressedBlocks - 1) * bytesPerBlock, bytesPerLastBlock);
        }

        _decompressedData = destination;
    }
    #pragma omp taskwait
    #pragma omp barrier
}

void Decompressor::decompressAdaptiveModel()
{
    decompressAdaptive();

    #pragma omp master
    {
        DEBUG_PRINT("Adaptive decompression with model");
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
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
            reverseDifferenceModel(source + (_numberOfCompressedBlocks - 1) * bytesPerBlock, destination + (_numberOfCompressedBlocks - 1) * bytesPerBlock, bytesPerLastBlock);
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
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
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
                switch (_bestBlockTraversals[i * _blocksPerRow + j])
                {
                case HORIZONTAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        deserializeBlock(source, destination, j, i);
                    }
                    break;
                
                case VERTICAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        transposeDeserializeBlock(source, destination, j, i);
                    }
                    break;

                default:
                    cerr << "Error: Unsupported traversal type" << endl;
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
    if (readInputFile(inputFileName, outputFileName))
    {
        #pragma omp parallel
        {
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
