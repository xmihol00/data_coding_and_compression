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
    switch (firstByte.getHeaderType() & MULTI_THREADED)
    {
        case SINGLE_THREADED:
            {
                DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_headerBuffer);
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsHeader));
            #ifdef _DEBUG_PRINT_ACTIVE_
                cerr << "Header: ";
                for (uint16_t i = 0; i < sizeof(DepthBitmapsHeader); i++)
                {
                    cerr << bitset<8>(_headerBuffer[i]) << " ";
                }
                cerr << endl;
            #endif
                _usedDepths = header.getCodeDepths();
                uint16_t bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                inputFile.read(reinterpret_cast<char *>(_symbolsAtDepths), bitmapsSize);
                fileSize -= sizeof(DepthBitmapsHeader) + bitmapsSize;
                _compressedSizesExScan[0] = 0;
            }
            break;

        case MULTI_THREADED:
            {
                DepthBitmapsMultiThreadedHeader &header = reinterpret_cast<DepthBitmapsMultiThreadedHeader &>(_headerBuffer);
                inputFile.read(reinterpret_cast<char *>(&header), sizeof(DepthBitmapsMultiThreadedHeader));
            
            #ifdef _DEBUG_PRINT_ACTIVE_
                cerr << "Header: ";
                for (uint16_t i = 0; i < sizeof(DepthBitmapsMultiThreadedHeader); i++)
                {
                    cerr << bitset<8>(_headerBuffer[i]) << " ";
                }
                cerr << endl;
            #endif

                _numberOfCompressedBlocks = header.getNumberOfThreads();
                _bitsPerCompressedBlockSize = header.getBitsPerBlockSize();
                _numberOfBytesCompressedBlocks = (_numberOfCompressedBlocks * _bitsPerCompressedBlockSize + 7) / 8;
                inputFile.read(reinterpret_cast<char *>(_rawBlockSizes), _numberOfBytesCompressedBlocks);
            
            #ifdef _DEBUG_PRINT_ACTIVE_
                cerr << "Compressed sizes: ";
                for (uint16_t i = 0; i < _numberOfBytesCompressedBlocks; i++)
                {
                    cerr << bitset<8>(reinterpret_cast<char *>(_rawBlockSizes)[i]) << " ";
                }
                cerr << endl;
            #endif

                _usedDepths = header.getCodeDepths();
                uint16_t bitmapsSize = sizeof(uint64v4_t) * popcount(_usedDepths);
                inputFile.read(reinterpret_cast<char *>(_symbolsAtDepths), bitmapsSize);
                fileSize -= sizeof(DepthBitmapsMultiThreadedHeader) + bitmapsSize + _numberOfBytesCompressedBlocks;
            }
            break;
        
        default:
            cerr << "Error: Unsupported header type" << endl;
            exit(1);
    }

#ifdef _DEBUG_PRINT_ACTIVE_
    cerr << "Symbols at depths: ";
    for (uint16_t i = 0; i < popcount(_usedDepths); i++)
    {
        cerr << "\n";
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
        cerr << bitset<64>(bits[0]) << " ";
        cerr << bitset<64>(bits[1]) << " ";
        cerr << bitset<64>(bits[2]) << " ";
        cerr << bitset<64>(bits[3]) << " ";
    }
    cerr << endl;
#endif
    
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
    #ifdef _DEBUG_PRINT_ACTIVE_
        cerr << "Block types: ";
        for (uint16_t i = 0; i < packedBlockCount; i++)
        {
            cerr << bitset<8>(_decompressionBuffer[i]) << " ";
        }
        cerr << endl;
    #endif

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

#ifdef _DEBUG_PRINT_ACTIVE_
    cerr << "Compressed data: ";
    for (uint16_t i = 0; i < 20; i++)
    {
        cerr << bitset<8>(_compressedData[i]) << " ";
    }
    cerr << endl;
#endif
    
    return true;
}

void Decompressor::parseBitmapHuffmanTree()
{
    uint8_t depthIdx = 0;
    uint8_t lastDepth = 0;
    uint16_t lastCode = -1;
    uint8_t masksIdx = 0;
    for (uint8_t i = 0; i < MAX_CODE_LENGTH; i++)
    {
        if (_usedDepths & (1UL << i))
        {
        #if __AVX512VPOPCNTDQ__
            uint64v4_t popCounts = _mm256_popcnt_epi64(_symbolsAtDepths[depthIdx]); // AVX512VPOPCNTDQ
            uint64_t *bits = reinterpret_cast<uint64_t *>(&popCounts);
            uint16_t sum1 = bits[0] + bits[1];
            uint16_t sum2 = bits[2] + bits[3];
            uint16_t symbolCount = sum1 + sum2;
        #else
            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + depthIdx);
            uint16_t symbolCount = _mm_popcnt_u64(bits[0]) + _mm_popcnt_u64(bits[1]) + _mm_popcnt_u64(bits[2]) + _mm_popcnt_u64(bits[3]);
        #endif
            uint8_t delta = i - lastDepth;
            lastCode = (lastCode + 1) << delta;

            _depthsIndices[masksIdx].depth = i;
            _depthsIndices[masksIdx].prefixLength = i - 16 + countl_zero(static_cast<uint16_t>(symbolCount - 1));
            uint16_t additionalPrefix = lastCode & (static_cast<uint16_t>(~0) >> (16 - _depthsIndices[masksIdx].depth + _depthsIndices[masksIdx].prefixLength));
            if (additionalPrefix)
            {
                _depthsIndices[masksIdx + 1].depth = i;
                _depthsIndices[masksIdx + 1].prefixLength = _depthsIndices[masksIdx].prefixLength;
                _depthsIndices[masksIdx].depth = i;
                _depthsIndices[masksIdx].prefixLength = lastDepth;
                _depthsIndices[masksIdx].symbolsAtDepthIndex = depthIdx;
                _depthsIndices[masksIdx].masksIndex = masksIdx;
                _indexPrefixLengths[masksIdx].index = masksIdx;
                _indexPrefixLengths[masksIdx].prefixLength = _depthsIndices[masksIdx].prefixLength;

                masksIdx++;

            }
            _depthsIndices[masksIdx].symbolsAtDepthIndex = depthIdx;
            _depthsIndices[masksIdx].masksIndex = masksIdx;
            _indexPrefixLengths[masksIdx].index = masksIdx;
            _indexPrefixLengths[masksIdx].prefixLength = _depthsIndices[masksIdx].prefixLength;


            masksIdx++;
            depthIdx++;
            lastCode += symbolCount - 1;
            lastDepth = i;
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
        _codePrefixesSmall[15 - _depthsIndices[i].masksIndex] = lastCode << (16 - _depthsIndices[i].depth);
        _codeMasksSmall[15 - _depthsIndices[i].masksIndex] = (~0U) << (16 - _depthsIndices[i].prefixLength);
        _prefixIndices[_depthsIndices[i].masksIndex] = symbolIdx;
        _prefixShifts[_depthsIndices[i].masksIndex] = _depthsIndices[i].prefixLength;
        _suffixShifts[_depthsIndices[i].masksIndex] = 16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength;
        symbolIdx += lastCode & (static_cast<uint16_t>(~0) >> (16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength));

        int16_t maxSymbols = 1 << (_depthsIndices[i].depth - _depthsIndices[i].prefixLength);
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

#if 0 // TODO
    #if __AVX512BW__ && __AVX512VL__
        reinterpret_cast<uint32v16_t *>(_indexPrefixLengthCodeCount)[0] = _mm512_setzero_si512();
        reinterpret_cast<uint32v16_t *>(_indexPrefixLengthCodeCount)[1] = _mm512_setzero_si512();
    #endif
    CodeLengthsSingleThreadedHeader &header = reinterpret_cast<CodeLengthsSingleThreadedHeader &>(_header);
    for (uint16_t i = 0, j = 0; i < NUMBER_OF_SYMBOLS / 2; i++, j += 2)
    {
        _codeLengthsSymbols[j].codeLength = header.codeLengths[i] >> 4;
        _codeLengthsSymbols[j].symbol = j;
        _indexPrefixLengthCodeCount[_codeLengthsSymbols[j].codeLength].codeCount++;
        _codeLengthsSymbols[j + 1].codeLength = header.codeLengths[i] & 0x0F;
        _codeLengthsSymbols[j + 1].symbol = j + 1;
        _indexPrefixLengthCodeCount[_codeLengthsSymbols[j + 1].codeLength].codeCount++;
    }

    uint8_t lastCodeCount = 0;
    for (uint8_t i = 0; i < MAX_CODE_LENGTH; i++)
    {
        _indexPrefixLengthCodeCount[i].cumulativeCodeCount = lastCodeCount;
        lastCodeCount += _indexPrefixLengthCodeCount[i].codeCount; 
        _indexPrefixLengths[i].index = i;
        _indexPrefixLengths[i].prefixLength = (i - 16 + countl_zero(static_cast<uint16_t>(_indexPrefixLengthCodeCount[i].codeCount - 1)));
    }
    sort(_indexPrefixLengths, _indexPrefixLengths + MAX_CODE_LENGTH, 
            [](const IndexPrefixLength &a, const IndexPrefixLength &b) { return a.prefixLength < b.prefixLength; });
    
    for (uint8_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
    {
        _indexPrefixLengthCodeCount[_indexPrefixLengths[i].index].prefixLength = _indexPrefixLengths[i].prefixLength;
        _indexPrefixLengthCodeCount[_indexPrefixLengths[i].index].index = i;
    }

    sort(_codeLengthsSymbols, _codeLengthsSymbols + NUMBER_OF_SYMBOLS, 
            [](const CodeLengthSymbol &a, const CodeLengthSymbol &b) { return a.codeLength < b.codeLength || (a.codeLength == b.codeLength && a.symbol < b.symbol); });

    uint16_t codesIdx = 0;
    while (_codeLengthsSymbols[codesIdx].codeLength == 0)
    {
        codesIdx++;
    }

    uint16_t lastCode = -1;
    uint8_t delta = 0;
    uint16_t symbolIdx = 0;
    for (uint8_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
    {
        if (_indexPrefixLengthCodeCount[i].prefixLength < 0)
        {
            delta++;
        }
        else
        {
            lastCode = (lastCode + 1) << delta;
            delta = 1;
            _codePrefixesSmall[_indexPrefixLengthCodeCount[i].index] = lastCode << (16 - i);
            _codeMasksSmall[_indexPrefixLengthCodeCount[i].index] = (~0U) << (16 - _indexPrefixLengthCodeCount[i].prefixLength);
            _prefixIndices[15 - _indexPrefixLengthCodeCount[i].index] = symbolIdx;
            _prefixShifts[15 - _indexPrefixLengthCodeCount[i].index] = _indexPrefixLengthCodeCount[i].prefixLength;
            _suffixShifts[15 - _indexPrefixLengthCodeCount[i].index] = 16 - i + _indexPrefixLengthCodeCount[i].prefixLength;
            symbolIdx += lastCode & (static_cast<uint16_t>(~0) >> (16 - i + _indexPrefixLengthCodeCount[i].prefixLength));
            for (uint16_t j = 0; j < _indexPrefixLengthCodeCount[i].codeCount; j++)
            {
                _symbolsTable[symbolIdx++] = _codeLengthsSymbols[codesIdx++].symbol;
            }
            lastCode += _indexPrefixLengthCodeCount[i].codeCount - 1;
        }
    }
#endif
    _codePrefixesSmallVector[0] = _mm256_and_si256(_codeMasksSmallVector[0], _codePrefixesSmallVector[0]);
    
    for (uint16_t i = 0; i < MAX_CODE_LENGTH; i++)
    {
        bitset<16> prefix(_codePrefixesSmall[i]);
        bitset<16> mask(_codeMasksSmall[i]);
        DEBUG_PRINT(i << ": " << prefix << " " << mask << " " << (int)_prefixShifts[i] << " " << (int)_suffixShifts[i] << " " << (int)_prefixIndices[i]);
    }
}

void Decompressor::parseThreadingInfo()
{
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
        DEBUG_PRINT("Block " << i << " size: " << _compressedSizes[i] << " ex scan: " << _compressedSizesExScan[i]);
    }
}

void Decompressor::parseHeader()
{
    switch (_header.getHeaderType() & MULTI_THREADED)
    {
        case MULTI_THREADED:
            parseThreadingInfo();
            [[fallthrough]];

        case SINGLE_THREADED:
            parseBitmapHuffmanTree();
            break;
        
        default:
            cout << "Error: Unsupported header type" << endl;
            exit(1);
    }
}

void Decompressor::transformRLE(uint16_t *compressedData, symbol_t *decompressedData, uint64_t bytesToDecompress)
{
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " bytesToDecompress: " << bytesToDecompress);
#if __AVX512BW__ && __AVX512VL__
    bool overflow;
    uint64_t decompressedIdx = 0;
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint16v16_t prefixes = _mm256_load_si256(_codePrefixesSmallVector);
    uint16v16_t masks = _mm256_load_si256(_codeMasksSmallVector);
    uint16_t current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
    DEBUG_PRINT("Current: " << bitset<16>(current));
    uint16v16_t currentVector = _mm256_set1_epi16(current);
    uint16v16_t masked = _mm256_and_si256(currentVector, masks);
    uint16v16_t xored = _mm256_xor_si256(prefixes, masked);
    uint16_t prefixBitMap = _mm256_cmp_epu16_mask(xored, _mm256_setzero_si256(), _MM_CMPINT_EQ);
    DEBUG_PRINT("PrefixBitMap: " << bitset<16>(prefixBitMap));
    uint8_t prefixIdx = countl_zero(prefixBitMap);
    DEBUG_PRINT("PrefixIdx: " << (int)prefixIdx);
    uint8_t prefixShift = _prefixShifts[prefixIdx];
    DEBUG_PRINT("PrefixLength: " << (int)prefixShift);
    uint8_t suffixShift = _suffixShifts[prefixIdx];
    DEBUG_PRINT("SuffixShift: " << (int)suffixShift);
    uint8_t suffix = static_cast<uint16_t>(current << prefixShift) >> suffixShift;
    uint8_t codeLength = 16 + prefixShift - suffixShift;
    DEBUG_PRINT("symbol idx: " << _prefixIndices[prefixIdx] + suffix << " " << (int)suffix);
    symbol_t symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
    decompressedData[decompressedIdx++] = symbol;
    DEBUG_PRINT("Symbol: " << (char)symbol << " " << (int)symbol << "\n");
    bitLength += codeLength;
    symbol_t lastSymbol = symbol;
    uint8_t sameSymbolCount = 0;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < bytesToDecompress)
    {
        //DEBUG_PRINT("BitLength: " << (int)bitLength);
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
        //DEBUG_PRINT("Current: " << bitset<16>(current));
        currentVector = _mm256_set1_epi16(current);
        masked = _mm256_and_si256(currentVector, masks);
        xored = _mm256_xor_si256(prefixes, masked);
        prefixBitMap = _mm256_cmp_epu16_mask(xored, _mm256_setzero_si256(), _MM_CMPINT_EQ);
        //DEBUG_PRINT("PrefixBitMap: " << bitset<16>(prefixBitMap));
        prefixIdx = countl_zero(prefixBitMap);
        //DEBUG_PRINT("PrefixIdx: " << (int)prefixIdx);
        prefixShift = _prefixShifts[prefixIdx];
        //DEBUG_PRINT("PrefixLength: " << (int)prefixShift);
        suffixShift = _suffixShifts[prefixIdx];
        //DEBUG_PRINT("SuffixShift: " << (int)suffixShift);
        suffix = static_cast<uint16_t>(current << prefixShift) >> suffixShift;
        codeLength = 16 + prefixShift - suffixShift;
        symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
        //DEBUG_PRINT("Suffix: " << (int)suffix);
        //DEBUG_PRINT("Thread: " << omp_get_thread_num() << " symbol: " << /*(char)symbol << " " <<*/ (int)symbol);
        decompressedData[decompressedIdx++] = symbol;
        sameSymbolCount += lastSymbol == symbol;
        sameSymbolCount >>= lastSymbol != symbol;
        lastSymbol = symbol;

        bitLength += codeLength;
        overflow = bitLength >= 16;
        bitLength &= 0x0F;
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
                uint32_t repetitions = ((current & 0x7FFF) >> BITS_PER_REPETITION_NUMBER);
                repetitions <<= multiplier;
                //DEBUG_PRINT("Thread: " << omp_get_thread_num() << " repetitions: " << repetitions);

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
            //DEBUG_PRINT("");
        }

        /*if (nextIdx >= 25)
        {
            exit(0);
        }*/
    }

    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " decompressedIdx: " << decompressedIdx << " block size:" << bytesToDecompress);
#endif
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
        for (uint32_t i = 0; i < 32; i++)
        {
            cerr << (int)_decompressedData[i] << " ";
        }
        cerr << endl;
        uint64_t bytesPerBlock = (_size + _numberOfCompressedBlocks - 1) / _numberOfCompressedBlocks;
        uint64_t bytesPerLastBlock = _size - bytesPerBlock * (_numberOfCompressedBlocks - 1);

        // schedule the decompression of each block as a task
        for (uint8_t i = 0; i < _numberOfCompressedBlocks - 1; i++)
        {
            #pragma omp task firstprivate(i)
            {
                reverseDifferenceModel(_decompressedData + i * bytesPerBlock, reinterpret_cast<symbol_t *>(_compressedData) + i * bytesPerBlock, bytesPerBlock);
            }
        }

        #pragma omp task // last block may be smaller
        {
            reverseDifferenceModel(_decompressedData + (_numberOfCompressedBlocks - 1) * bytesPerBlock, reinterpret_cast<symbol_t *>(_compressedData) + (_numberOfCompressedBlocks - 1) * bytesPerBlock, bytesPerLastBlock);
        }
    }
    #pragma omp taskwait
    #pragma omp barrier

    #pragma omp master
    {
        _decompressedData = reinterpret_cast<symbol_t *>(_compressedData);
        for (uint32_t i = 0; i < 32; i++)
        {
            cerr << (int)_decompressedData[i] << " ";
        }
        cerr << endl;
    }
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
                        deserializeBlock(_decompressionBuffer, destination, j, i);
                    }
                    break;
                
                case VERTICAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        transposeDeserializeBlock(_decompressionBuffer, destination, j, i);
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
        parseHeader();

        #pragma omp parallel
        {
            if (_header.getHeaderType() & MODEL)
            {
                if (_header.getHeaderType() & ADAPTIVE)
                {
                    // TODO
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
