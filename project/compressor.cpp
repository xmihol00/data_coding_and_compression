#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint64_t width, int32_t numberOfThreads)
    : HuffmanRLECompression(model, adaptive, width, numberOfThreads) { }

Compressor::~Compressor()
{ 
    if (_sourceBuffer != nullptr)
    {
        free(_sourceBuffer);
    }

    for (uint8_t i = 0; i < MAX_NUM_THREADS; i++)
    {
        if (_rlePerBlockCounts[i] != nullptr)
        {
            delete[] _rlePerBlockCounts[i];
        }
    }

    if (_destinationBuffer != nullptr)
    {
        free(_destinationBuffer);
    }
}

void Compressor::clearMemory()
{
    uint64_t *repetitions = _repetitions;
    #pragma omp simd aligned(repetitions: 64) simdlen(8)
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        repetitions[i] = 0;
    }
    _compressionUnsuccessful = false;
}

void Compressor::computeHistogram()
{
    DEBUG_PRINT("Computing histogram with thread " << omp_get_thread_num() << " started");
    int threadId = omp_get_thread_num();

    uint64_t *intHistogram = _intHistogram;
    symbol_t *buffer = _sourceBuffer;
    uint16_t numberOfThreads = _numberOfThreads > 8 ? 8 : _numberOfThreads;

    if (threadId < 8)
    {
        uint64_t symbolsToProcess = (_size + 7) / numberOfThreads;
        intHistogram += threadId * NUMBER_OF_SYMBOLS;
        buffer += threadId * symbolsToProcess;
        if (threadId == _numberOfThreads - 1)
        {
            symbolsToProcess = _size - threadId * symbolsToProcess;
        }

        #pragma omp simd aligned(intHistogram: 64) simdlen(8)
        for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            intHistogram[i] = 0;
        }

        uint8_t lastSymbol = ~buffer[0];
        uint32_t repetitions = 0;
        for (uint64_t i = 0; i < symbolsToProcess; i++)
        {
            if (buffer[i] != lastSymbol)
            {
                intHistogram[buffer[i]]++;

                #pragma omp atomic
                _repetitions[repetitions]++;

                repetitions = 0;
            }
            else
            {
                repetitions++;
                if (repetitions == 255)
                {
                    #pragma omp atomic
                    _repetitions[repetitions]++;

                    repetitions = 0;
                }
            }
            lastSymbol = buffer[i];
        }
    }
    #pragma omp barrier

    #pragma omp single
    {
        intHistogram = _intHistogram;
        for (uint8_t i = 1; i < numberOfThreads; i++)
        {
            #pragma omp simd aligned(intHistogram: 64) simdlen(8)
            for (uint16_t j = 0; j < NUMBER_OF_SYMBOLS; j++)
            {
                intHistogram[j] += intHistogram[i * NUMBER_OF_SYMBOLS + j];
            }
        }

        FrequencySymbolIndex *structHistogram = _structHistogram;
        #pragma omp simd aligned(structHistogram, intHistogram: 64) simdlen(8)
        for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            reinterpret_cast<uint64_t *>(structHistogram)[i] = intHistogram[i] << 8;
            structHistogram[i].index = i;
            structHistogram[i] = intHistogram[i] == 0 ? MAX_FREQUENCY_SYMBOL_INDEX : structHistogram[i];
        }
    }
    // implicit barrier

    /*uint64_t maxFrequency = max_element(histogram, histogram + NUMBER_OF_SYMBOLS)[0];
    if (maxFrequency >= 1UL << 40) // this will likely never happen 
    {
        cerr << "Error: Input file is too large." << endl;
        exit(1);
    }*/

    DEBUG_PRINT("Histogram computed");
}

void Compressor::buildHuffmanTree()
{
    DEBUG_PRINT("Building Huffman tree");
    uint32_t *symbolsParentsDepths = reinterpret_cast<uint32_t *>(_symbolsParentsDepths);
    uint16_t *parentsSortedIndices = _parentsSortedIndices;
    DEBUG_PRINT("Size: " << sizeof(FrequencySymbolIndex) << " " << sizeof(SymbolParentDepth) << " " << sizeof(uint32_t) << " " << sizeof(uint16_t));

    DEBUG_PRINT("Clearing symbolsParentsDepths and parentsSortedIndices");
    #pragma omp simd aligned(symbolsParentsDepths: 64) simdlen(16)
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        symbolsParentsDepths[i] = 0;
    }
    #pragma omp simd aligned(parentsSortedIndices: 64) simdlen(32)
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        parentsSortedIndices[i] = 0;
    }
    DEBUG_PRINT("Cleared symbolsParentsDepths and parentsSortedIndices");
    
    uint16_t sortedIdx = 0;
    union Minimum
    {
        uint64_t intMin;
        FrequencySymbolIndex structMin;
    };
    
    Minimum firstMin;
    Minimum secondMin;
    
    DEBUG_PRINT("Starting sorting symbols");
    while (true)
    {
        uint64v8_t min1 = _vectorHistogram[0];
        uint64v8_t min2 = _vectorHistogram[1];
        for (uint8_t i = 2; i < NUMBER_OF_SYMBOLS / 8 ; i += 2)
        {
            uint64v8_t current1 = _vectorHistogram[i];
            uint64v8_t current2 = _vectorHistogram[i + 1];
            
            min1 = _mm512_min_epi64(min1, current1);
            min2 = _mm512_min_epi64(min2, current2);
        }
        min1 = _mm512_min_epi64(min1, min2);
        firstMin.intMin = _mm512_reduce_min_epi64(min1);
        _structHistogram[firstMin.structMin.index] = MAX_FREQUENCY_SYMBOL_INDEX;
        
        min1 = _vectorHistogram[0];
        min2 = _vectorHistogram[1];
        for (uint8_t i = 2; i < NUMBER_OF_SYMBOLS / 8 ; i += 2)
        {
            uint64v8_t current1 = _vectorHistogram[i];
            uint64v8_t current2 = _vectorHistogram[i + 1];
            min1 = _mm512_min_epi64(min1, current1);
            min2 = _mm512_min_epi64(min2, current2);
        }
        min1 = _mm512_min_epi64(min1, min2);
        
        secondMin.intMin = _mm512_reduce_min_epi64(min1);

        uint16_t firstIndex = _parentsSortedIndices[firstMin.structMin.index];
        sortedIdx = firstIndex == 0 ? sortedIdx + 1 : sortedIdx;
        firstIndex = firstIndex == 0 ? sortedIdx : firstIndex;

        uint16_t secondIndex = _parentsSortedIndices[secondMin.structMin.index];
        sortedIdx = secondIndex == 0 ? sortedIdx + 1 : sortedIdx;
        secondIndex = secondIndex == 0 ? sortedIdx : secondIndex;

        sortedIdx++;

        _symbolsParentsDepths[firstIndex].depth = _parentsSortedIndices[firstMin.structMin.index] == 0;
        _symbolsParentsDepths[firstIndex].parent = sortedIdx;
        _symbolsParentsDepths[firstIndex].symbol = firstMin.structMin.index;
        firstMin.structMin.index = 0;

        if (secondMin.intMin == INT64_MAX)
        {
            _symbolsParentsDepths[sortedIdx].depth = 0;
            sortedIdx -= 2;
            break;
        }

        _symbolsParentsDepths[secondIndex].depth = _parentsSortedIndices[secondMin.structMin.index] == 0;
        _symbolsParentsDepths[secondIndex].parent = sortedIdx;
        _symbolsParentsDepths[secondIndex].symbol = secondMin.structMin.index;
        secondIndex = secondMin.structMin.index;
        secondMin.structMin.index = 0;

        Minimum nextNode;
        nextNode.intMin = firstMin.intMin + secondMin.intMin;
        nextNode.structMin.index = secondIndex;
        _structHistogram[secondIndex] = nextNode.structMin;
        _parentsSortedIndices[secondIndex] = sortedIdx;
    }
    DEBUG_PRINT("Symbols sorted");

    DEBUG_PRINT("Building the tree");
    int16_t symbolsDepthsIdx = 0;
    for (uint16_t i = sortedIdx; i > 0; i--)
    {
        uint8_t nextDepth = _symbolsParentsDepths[_symbolsParentsDepths[i].parent].depth + 1;
        if (_symbolsParentsDepths[i].depth)
        {
            _symbols[symbolsDepthsIdx] = _symbolsParentsDepths[i].symbol;
            _depths[symbolsDepthsIdx] = nextDepth;
            symbolsDepthsIdx++;
        }
        _symbolsParentsDepths[i].depth = nextDepth;
    }
    DEBUG_PRINT("Symbols and depths sorted");

    _numberOfSymbols = symbolsDepthsIdx;
    symbolsDepthsIdx--;

    DEBUG_PRINT("Pruning the tree");
    uint8_t maxDepth = _depths[symbolsDepthsIdx];
    constexpr uint16_t decreasedMaxCodeLength = MAX_CODE_LENGTH - 1;
    bool balanced = maxDepth > decreasedMaxCodeLength;
    if (maxDepth > decreasedMaxCodeLength)
    {
        int16_t depthIdx = symbolsDepthsIdx;
        uint8_t maxDepthShift = maxDepth - decreasedMaxCodeLength;
        int32_t debt = 0;
        while (_depths[depthIdx] > decreasedMaxCodeLength)
        {
            uint8_t maxDepthDifference = _depths[depthIdx] - decreasedMaxCodeLength;
            debt += ((1 << (maxDepthDifference)) - 1) << (maxDepthShift - maxDepthDifference);
            _depths[depthIdx] = decreasedMaxCodeLength;
            depthIdx--;
        }

        while (_depths[depthIdx] == decreasedMaxCodeLength) // skip symbols already at max depth
        {
            depthIdx--;
        }

        // repay the debt
        while (debt > 0)
        {
            debt -= 1 << (maxDepth - _depths[depthIdx] - 1);
            _depths[depthIdx]++;
            depthIdx--;
        }

        // steal back overpaid debt
        for (int16_t i = depthIdx + 2; i < symbolsDepthsIdx; i++)
        {
            int32_t stealBack = 1 << (maxDepth - _depths[i]);
            if (stealBack + debt <= 0)
            {
                _depths[i]--;
                debt += stealBack;
            }
            else
            {
                break;
            }
        }      
    }
    DEBUG_PRINT("Tree pruned");
    
    DEBUG_PRINT("Rebalancing the tree");
    // rebalance the tree
    uint16_t depthIdx = 0;
    uint16_t nodesToRebalance = 1 << (_depths[0] - 1);
    uint8_t depth = _depths[0];
    DEBUG_PRINT("First depth: " << (int)_depths[0]);
    while (depthIdx < _numberOfSymbols)
    {
        nodesToRebalance = depthIdx + nodesToRebalance < _numberOfSymbols ? nodesToRebalance : _numberOfSymbols - depthIdx;
        for (uint16_t i = 0; i < nodesToRebalance; i++)
        {
            _depths[depthIdx++] = depth;
        }

        if (depth < _depths[depthIdx])
        {
            DEBUG_PRINT("Next depth: " << (int)_depths[depthIdx] << " current depth: " << (int)depth);
            nodesToRebalance <<= _depths[depthIdx] - depth;
            depth = _depths[depthIdx];
        }
        else if (depth == _depths[depthIdx])
        {
            DEBUG_PRINT("Same depths depth: " << (int)_depths[depthIdx] << " current depth: " << (int)depth);
            nodesToRebalance <<= 1;
            depth += 1;
        }

        depth++;
    }
    DEBUG_PRINT("Tree rebalanced");
    DEBUG_PRINT("Last depth: " << (int)_depths[_numberOfSymbols - 1]);

    DEBUG_PRINT("Huffman tree built");
}

void Compressor::populateCodeTable()
{
    DEBUG_PRINT("Populating code table");

    // clear the code table
    uint32_t *codeTable = reinterpret_cast<uint32_t *>(_codeTable);
    #pragma omp simd aligned(codeTable: 64) simdlen(16)
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        codeTable[i] = 0;
    }

    // clear symbols depth bitmaps
    uint32_t *symbolsAtDepths = reinterpret_cast<uint32_t *>(_symbolsAtDepths);
    #pragma omp simd aligned(symbolsAtDepths: 64) simdlen(16)
    for (uint16_t i = 0; i < MAX_CODE_LENGTH * 8; i++)
    {
        symbolsAtDepths[i] = 0;
    }

    // sort symbols at depths
    _usedDepths = 0;
    for (uint16_t i = 0; i < _numberOfSymbols; i++)
    {
        uint64_t bits[4];
        reinterpret_cast<uint64v4_t *>(bits)[0] = _mm256_setzero_si256();
        _usedDepths |= 1U << _depths[i];
        uint8_t symbol = 255 - _symbols[i];
        bits[3] = (uint64_t)(symbol < 64) << symbol;
        bits[2] = (uint64_t)(symbol < 128 && symbol >= 64) << (symbol - 64);
        bits[1] = (uint64_t)(symbol < 192 && symbol >= 128) << (symbol - 128);
        bits[0] = (uint64_t)(symbol >= 192) << (symbol - 192);
        _symbolsAtDepths[_depths[i]] = _mm256_or_si256(_symbolsAtDepths[_depths[i]], reinterpret_cast<uint64v4_t *>(bits)[0]);
    }
    DEBUG_PRINT("Depths used: " << bitset<32>(_usedDepths));

    uint16_t lastCode = -1;
    uint8_t lastDepth = 0;
    uint8_t depthIndex = 0;
    uint8_t delta = 0;
    // populate the code table
    for (uint8_t i = 0; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i))
        {
            _symbolsAtDepths[depthIndex++] = _symbolsAtDepths[i];
            DEBUG_PRINT("Depth: " << (int)i);
            uint64v4_t bitsVector = _mm256_load_si256(_symbolsAtDepths + i);
            uint64_t *bits = reinterpret_cast<uint64_t *>(&bitsVector);
            lastCode = (lastCode + 1) << delta;
            uint16_t oldCode = lastCode;
            lastCode--;
            for (uint8_t j = 0; j < 4; j++)
            {
                uint8_t symbol = j << 6;
                uint8_t leadingZeros;
                while ((leadingZeros = countl_zero(bits[j])) < 64)
                {
                    uint8_t adjustedSymbol = symbol + leadingZeros;
                    lastCode++;
                    DEBUG_PRINT("Code: " << bitset<16>(lastCode) << " Symbol: " << (int)adjustedSymbol);
                    bits[j] ^= 1UL << (63 - leadingZeros); // remove the current symbol from the bitmap
                    _codeTable[adjustedSymbol].code = lastCode;
                    _codeTable[adjustedSymbol].length = i;
                }
            }
            delta = 0;
        }
        delta++;
    }

    DEBUG_PRINT("Code table populated");
}

void Compressor::transformRLE(symbol_t *sourceData, uint16_t *compressedData, uint32_t &compressedSize, uint64_t &startingIdx)
{
    int threadNumber = omp_get_thread_num();
    DEBUG_PRINT("Transforming RLE with thread " << threadNumber << " started");

    uint32_t bytesToCompress = (_size + _numberOfThreads - 1) / _numberOfThreads;
    bytesToCompress += bytesToCompress & 0x1; // ensure even number of bytes per thread, TODO solve single thread case
    startingIdx = bytesToCompress * threadNumber;
    symbol_t firstSymbol = sourceData[startingIdx];
    if (threadNumber == _numberOfThreads - 1 && _numberOfThreads > 1) // last thread
    {
        // ensure last symbol in a block is different from the first in next block
        sourceData[startingIdx] = ~sourceData[startingIdx - 1];
        sourceData[_size] = ~sourceData[_size - 1];

        bytesToCompress -= bytesToCompress * _numberOfThreads - _size; // ensure last thread does not run out of bounds
    }
    else if (threadNumber != 0) // remaining threads apart from the first one
    {
        sourceData[startingIdx] = ~sourceData[startingIdx - 1];
    }
    else // single threaded execution
    {
        sourceData[_size] = ~sourceData[_size - 1];
    }
    sourceData += startingIdx;
    DEBUG_PRINT("Starting index: " << startingIdx << " First symbol: " << (char)firstSymbol);
    startingIdx += _threadPadding * threadNumber;
    DEBUG_PRINT("Starting index with padding: " << startingIdx);
    compressedData += startingIdx >> 1;

    #pragma omp barrier // synchronize

    compressedData[0] = 0;
    uint32_t currentCompressedIdx = 0;
    uint32_t nextCompressedIdx = 1;
    uint32_t sourceDataIdx = 1;
    uint32_t sameSymbolCount = 1;
    symbol_t current = firstSymbol;
    uint8_t chunkIdx = 0;

    uint64_t shortsToCompress = (bytesToCompress + _threadPadding - 2) >> 1;
    while (sourceDataIdx <= bytesToCompress && nextCompressedIdx < shortsToCompress)
    {
        if (sourceData[sourceDataIdx] == current)
        {
            sameSymbolCount++;
        }
        else
        {
            uint16_t codeLength = _codeTable[current].length;
            uint16_t code = _codeTable[current].code << (16 - codeLength);
            //bitset<16> maskedCodeBits(code);
            //DEBUG_PRINT((char)current << ": " << maskedCodeBits << ", times:" << sameSymbolCount);

            for (uint32_t i = 0; i < sameSymbolCount && i < 3; i++)
            {
                uint16_t downShiftedCode = code >> chunkIdx;
                uint16_t upShiftedCode = code << (16 - chunkIdx);
                compressedData[currentCompressedIdx] |= downShiftedCode;
                compressedData[nextCompressedIdx] = 0;
                compressedData[nextCompressedIdx] |= upShiftedCode;
                chunkIdx += codeLength;
                bool moveChunk = chunkIdx >= 16;
                chunkIdx &= 15;
                currentCompressedIdx += moveChunk;
                nextCompressedIdx += moveChunk;
            }

            bool repeating = sameSymbolCount >= 3;
            sameSymbolCount -= 3;
            while (repeating)
            {
                uint16_t count = sameSymbolCount & ((~0U) >> (16 - BITS_PER_REPETITION_NUMBER));
                sameSymbolCount >>= (BITS_PER_REPETITION_NUMBER - 1);
                repeating = sameSymbolCount > 0;
                count |= repeating << (BITS_PER_REPETITION_NUMBER - 1);
                count <<= (16 - BITS_PER_REPETITION_NUMBER);
                uint16_t downShiftedCount = count >> chunkIdx;
                uint16_t upShiftedCount = count << (16 - chunkIdx);
                compressedData[currentCompressedIdx] |= downShiftedCount;
                compressedData[nextCompressedIdx] = 0;
                compressedData[nextCompressedIdx] |= upShiftedCount;
                chunkIdx += BITS_PER_REPETITION_NUMBER;
                bool moveChunk = chunkIdx >= 16;
                chunkIdx &= 15;
                currentCompressedIdx += moveChunk;
                nextCompressedIdx += moveChunk;
            }

            sameSymbolCount = 1;
            current = sourceData[sourceDataIdx];
        }
        sourceDataIdx++;
    }
    
    _compressionUnsuccessful = sourceDataIdx <= bytesToCompress;
    compressedSize = nextCompressedIdx * 2;
    DEBUG_PRINT("RLE transformed, thread " << omp_get_thread_num() << " compressed size: " << compressedSize);
}

void Compressor::createHeader()
{
    DEBUG_PRINT("Creating header");

    if (_numberOfThreads > 1)
    {
        DEBUG_PRINT("Multi-threaded header with: " << _numberOfThreads << " threads");
        uint16_t maxBitsCompressedSizes = 0;
        for (uint8_t i = 0; i < _numberOfThreads; i++)
        {
            uint16_t bits = 32 - countl_zero(_compressedSizes[i]);
            maxBitsCompressedSizes = max(maxBitsCompressedSizes, bits);
        }
        DEBUG_PRINT("Max bits for compressed sizes: " << maxBitsCompressedSizes);

        uint8_t *packedCompressedSizes = reinterpret_cast<uint8_t *>(_compressedSizes);
        uint16_t idx = 0;
        uint8_t chunkIdx = 0;
        for (uint8_t i = 0; i < _numberOfThreads; i++)
        {
            uint32_t compressedSize = _compressedSizes[i];
            uint8_t bits = maxBitsCompressedSizes;
            compressedSize <<= 32 - maxBitsCompressedSizes;
            packedCompressedSizes[idx] = i == 0 ? 0 : packedCompressedSizes[idx];

            do
            {
                packedCompressedSizes[idx] |= compressedSize >> (24 + chunkIdx);
                uint8_t storedBits = bits < 8 - chunkIdx ? bits : 8 - chunkIdx;
                compressedSize <<= storedBits;
                bits -= storedBits;
                chunkIdx += storedBits;
                if (chunkIdx >= 8)
                {
                    chunkIdx &= 0x7;
                    packedCompressedSizes[++idx] = 0;
                }
            } 
            while (bits);
        }

        reinterpret_cast<uint16v16_t *>(_headerBuffer)[0] = _mm256_setzero_si256(); // clear the buffer
        DepthBitmapsMultiThreadedHeader &header = reinterpret_cast<DepthBitmapsMultiThreadedHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.insertHeaderType(MULTI_THREADED);
        header.setVersion(0);
        header.setNumberOfThreads(_numberOfThreads);
        header.setBitsPerBlockSize(maxBitsCompressedSizes);
        header.setCodeDepths(_usedDepths);
        DEBUG_PRINT("Code depths: " << bitset<16>(_usedDepths));
        _headerSize = sizeof(DepthBitmapsMultiThreadedHeader);
        _threadBlocksSizesSize = (_numberOfThreads * maxBitsCompressedSizes + 7) / 8;
    }
    else
    {
        DEBUG_PRINT("Single-threaded header");
        reinterpret_cast<uint16v16_t *>(_headerBuffer)[0] = _mm256_setzero_si256(); // clear the buffer
        DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.insertHeaderType(SINGLE_THREADED);
        header.setVersion(0);
        header.setCodeDepths(_usedDepths);
        DEBUG_PRINT("Code depths: " << bitset<16>(_usedDepths));
        _headerSize = sizeof(DepthBitmapsHeader);
        _threadBlocksSizesSize = 0;
    }

    BasicHeader &header = reinterpret_cast<BasicHeader &>(_headerBuffer);
    if (_adaptive)
    {
        header.insertHeaderType(ADAPTIVE);
        header.setWidth(_width);
        header.setHeight(_height);

        _blockTypesByteSize = (_blockCount * BITS_PER_BLOCK_TYPE + 7) / 8;
        DEBUG_PRINT("Block types byte size: " << _blockTypesByteSize);
        _blockTypes = new uint8_t[_blockTypesByteSize];

        for (uint32_t i = 0, j = 0; j < _blockCount; i++, j += 4)
        {
            _blockTypes[i] = 0;
            _blockTypes[i] |= _bestBlockTraversals[j] << 6;
            _blockTypes[i] |= _bestBlockTraversals[j + 1] << 4;
            _blockTypes[i] |= _bestBlockTraversals[j + 2] << 2;
            _blockTypes[i] |= _bestBlockTraversals[j + 3];
        }
    }
    else
    {
        header.insertHeaderType(STATIC);
        header.setSize(_size);
    }
    
    if (_model)
    {
        header.insertHeaderType(MODEL);
    }
    else
    {
        header.insertHeaderType(DIRECT);
    }
    
    DEBUG_PRINT("Header created");
}

void Compressor::writeOutputFile(string outputFileName, string inputFileName)
{
    DEBUG_PRINT("Writing output file");

    ofstream outputFile(outputFileName, ios::binary);
    if (!outputFile.is_open())
    {
        cerr << "Error: Unable to open output file '" << outputFileName << "'." << endl;
        exit(1);
    }
    
    if ((_size < _headerSize + _threadBlocksSizesSize + sizeof(uint64v4_t) * popcount(_usedDepths) + _compressedSizesExScan[_numberOfThreads]) ||
        _compressionUnsuccessful) // data not compressed
    {
        DEBUG_PRINT("Compressed data is larger than the input file");
        FirstByteHeader &header = reinterpret_cast<FirstByteHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.setNotCompressed();
        outputFile.write(reinterpret_cast<char *>(&_headerBuffer), sizeof(FirstByteHeader));

        ifstream inputFile(inputFileName, ios::binary);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << inputFileName << "'." << endl;
            exit(1);
        }
        inputFile.read(reinterpret_cast<char *>(_sourceBuffer), _size);   
        inputFile.close();

        outputFile.write(reinterpret_cast<char *>(_sourceBuffer), _size);
    }
    else // data successfully compressed
    {
        outputFile.write(reinterpret_cast<char *>(&_headerBuffer), _headerSize);
        outputFile.write(reinterpret_cast<char *>(_compressedSizes), _threadBlocksSizesSize);
        outputFile.write(reinterpret_cast<char *>(_symbolsAtDepths), sizeof(uint64v4_t) * popcount(_usedDepths));
        outputFile.write(reinterpret_cast<char *>(_blockTypes), _blockTypesByteSize);
        outputFile.write(reinterpret_cast<char *>(_destinationBuffer), _compressedSizesExScan[_numberOfThreads]); 

    #ifdef _DEBUG_PRINT_ACTIVE_ && 0
        cerr << "Header: ";
        for (uint16_t i = 0; i < _headerSize; i++)
        {
            cerr << bitset<8>(_headerBuffer[i]) << " ";
        }
        cerr << endl;

        cerr << "Compressed sizes: ";
        for (uint16_t i = 0; i < _threadBlocksSizesSize; i++)
        {
            cerr << bitset<8>(reinterpret_cast<char *>(_compressedSizes)[i]) << " ";
        }
        cerr << endl;

        uint16_t codes = popcount(_usedDepths);
        DEBUG_PRINT("Used depths: " << bitset<32>(_usedDepths));
        DEBUG_PRINT("Number of used depths: " << codes);
        cerr << "Symbols at depths: ";
        for (uint16_t i = 0; i < codes; i++)
        {
            cerr << "\n" << i << ": ";
            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
            cerr << bitset<64>(bits[0]) << " ";
            cerr << bitset<64>(bits[1]) << " ";
            cerr << bitset<64>(bits[2]) << " ";
            cerr << bitset<64>(bits[3]) << " ";
        }
        cerr << endl;

        cerr << "Block types: ";
        for (uint16_t i = 0; i < _blockTypesByteSize; i++)
        {
            cerr << bitset<8>(_blockTypes[i]) << " ";
        }
        cerr << endl;

        cerr << "Compressed data: ";
        for (uint16_t i = 0; i < 10; i++)
        {
            cerr << bitset<16>(reinterpret_cast<uint16_t *>(_destinationBuffer)[i]) << " ";
        }
        cerr << endl;
    #endif
    }
    outputFile.close();

    DEBUG_PRINT("Output file written");
}

void Compressor::readInputFile(string inputFileName)
{
    ifstream inputFile(inputFileName, ios::binary);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file '" << inputFileName << "'." << endl;
        exit(1);
    }

    // get the size of the input file
    inputFile.seekg(0, ios::end);
    _size = inputFile.tellg();
    inputFile.seekg(0, ios::beg);

    if (_size > UINT32_MAX) // TODO: possibly allow larger files
    {
        cerr << "Error: The input file is too large." << endl;
        exit(1);
    }
    else if (_size == 0) // TODO: possibly allow empty files
    {
        cerr << "Error: The input file is empty. Nothing to compress here." << endl;
        exit(1);
    }

    _height = _size / _width;
    if (_height * _width != _size) // verify that file is rectangular
    {
        cerr << "Error: The input file size is not a multiple of the specified width." << endl;
        exit(1);
    }

    uint64_t roundedSize = ((_size + _numberOfThreads - 1) / _numberOfThreads) * _numberOfThreads + 1;
    DEBUG_PRINT("Rounded size: " << roundedSize);
    _threadPadding = _numberOfThreads > 1 ? (roundedSize >> 4) : 0;
    DEBUG_PRINT("Thread padding: " << _threadPadding);
    roundedSize += _threadPadding * _numberOfThreads;
    _sourceBuffer = static_cast<symbol_t *>(aligned_alloc(64, roundedSize * sizeof(symbol_t)));
    _destinationBuffer = static_cast<symbol_t *>(aligned_alloc(64, roundedSize * sizeof(symbol_t)));
    if (_sourceBuffer == nullptr || _destinationBuffer == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the input file." << endl;
        exit(1);
    }
    cerr << (void *) _sourceBuffer << " " << (void *) _destinationBuffer << endl;

    inputFile.read(reinterpret_cast<char *>(_sourceBuffer), _size);
    inputFile.close();
}

void Compressor::compressStatic()
{
    int threadNumber = omp_get_thread_num();
    computeHistogram(); // TODO: parallelize

    #pragma omp single
    {
        DEBUG_PRINT("Static compression started");
      
        buildHuffmanTree();

        populateCodeTable();
    } // implicit barrier

    // all threads compress their part of the data, data is decomposed between threads based on their number/ID
    uint64_t startingIdx;
    {
        transformRLE(_sourceBuffer, reinterpret_cast<uint16_t *>(_destinationBuffer), _compressedSizes[threadNumber], startingIdx);
    }
    #pragma omp barrier

    #pragma omp single
    {
        // exclusive scan of compressed sizes
        _compressedSizesExScan[0] = 0;
        for (int32_t i = 0; i < _numberOfThreads; i++)
        {
            _compressedSizesExScan[i + 1] = _compressedSizesExScan[i] + _compressedSizes[i];
        }

        DEBUG_PRINT("Total compressed size: " << _compressedSizesExScan[_numberOfThreads]);
    } // implicit barrier

    // all threads pack the compressed data back into the initial buffer
    {
        copy(_destinationBuffer + startingIdx, _destinationBuffer + startingIdx + _compressedSizes[threadNumber], _sourceBuffer + _compressedSizesExScan[threadNumber]);
    }
    #pragma omp barrier

    #pragma omp master 
    {
        swap(_sourceBuffer, _destinationBuffer);
        createHeader();
        
        DEBUG_PRINT("Static compression finished");
    }
}

void Compressor::analyzeImageAdaptive()
{
    #pragma omp sections
    {
        #pragma omp section // histogram of symbols, same for each traversal technique
        {
            computeHistogram();
            //sort(_intHistogram, _intHistogram + NUMBER_OF_SYMBOLS, 
            //     [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
        }

        #pragma omp section // horizontal lines in blocks
        {
            constexpr AdaptiveTraversals sectionId = HORIZONTAL;
            _rlePerBlockCounts[sectionId] = new int32_t[_blockCount];
            int32_t *rleCounts = _rlePerBlockCounts[sectionId];

            uint32_t blockIdx = 0;
            for (uint32_t i = 0; i < _width; i += BLOCK_SIZE) // TODO: solve not divisible by BLOCK_SIZE
            {
                uint32_t firstRow = i * _width;
                for (uint32_t j = 0; j < _height; j += BLOCK_SIZE)
                {
                    uint32_t firstColumn = firstRow + j;
                    uint16_t lastSymbol = -1;
                    int32_t sameSymbolCount = -3;
                    int32_t rleCount = 0;
                    for (uint32_t k = 0; k < BLOCK_SIZE; k++)
                    {
                        for (uint32_t l = 0; l < BLOCK_SIZE; l++)
                        {
                            bool sameSymbol = _sourceBuffer[firstColumn + l] == lastSymbol;
                            lastSymbol = _sourceBuffer[firstColumn + l];
                        #if 0
                            rleCount += sameSymbolCount * (!sameSymbol && sameSymbolCount >= -1);
                            sameSymbolCount = (sameSymbolCount + sameSymbol) * sameSymbol + (-3 * !sameSymbol);
                        #endif
                            rleCount += !sameSymbol && sameSymbolCount >= -1 ? sameSymbolCount : 0;
                            sameSymbolCount++;
                            sameSymbolCount = sameSymbol ? sameSymbolCount : -3;
                        }

                        firstColumn += _width;
                    }
                    rleCounts[blockIdx++] = rleCount;
                }
            }
        }

        #pragma omp section // vertical lines in blocks
        {
            constexpr AdaptiveTraversals sectionId = VERTICAL;
            _rlePerBlockCounts[sectionId] = new int32_t[_blockCount];
            int32_t *rleCounts = _rlePerBlockCounts[sectionId];
            
            int32_t blockIdx = 0;
            for (uint32_t i = 0; i < _width; i += BLOCK_SIZE) // TODO: solve not divisible by BLOCK_SIZE
            {
                uint32_t firstRow = i * _width;
                for (uint32_t j = 0; j < _height; j += BLOCK_SIZE)
                {
                    uint32_t firstColumn = firstRow + j;
                    uint16_t lastSymbol = -1;
                    int32_t sameSymbolCount = -3;
                    int32_t rleCount = 0;
                    for (uint32_t k = 0; k < BLOCK_SIZE; k++)
                    {
                        uint32_t column = firstColumn + k;
                        for (uint32_t l = 0; l < BLOCK_SIZE; l++)
                        {
                            bool sameSymbol = _sourceBuffer[column] == lastSymbol;
                            lastSymbol = _sourceBuffer[column];
                        #if 0
                            rleCount += sameSymbolCount * (!sameSymbol && sameSymbolCount >= -1);
                            sameSymbolCount = (sameSymbolCount + sameSymbol) * sameSymbol + (-3 * !sameSymbol);
                        #endif
                            rleCount += !sameSymbol && sameSymbolCount >= -1 ? sameSymbolCount : 0;
                            sameSymbolCount++;
                            sameSymbolCount = sameSymbol ? sameSymbolCount : -3;

                            column += _width;   
                        }
                    }
                    rleCounts[blockIdx++] = rleCount;
                }
            }
        }
    }
}

void Compressor::applyDiferenceModel(symbol_t *source, symbol_t *destination)
{
    int threadNumber = omp_get_thread_num();
    uint64_t bytesToProcess = (_size + _numberOfThreads - 1) / _numberOfThreads;
    bytesToProcess += bytesToProcess & 0x1; // ensure even number of bytes per thread, TODO solve single thread case
    uint64_t startingIdx = bytesToProcess * threadNumber;
    if (threadNumber == _numberOfThreads - 1 && _numberOfThreads > 1) // last thread
    {
        bytesToProcess -= bytesToProcess * _numberOfThreads - _size; // ensure last thread does not run out of bounds
    }
    source += startingIdx;
    destination += startingIdx;

    destination[0] = source[0];

    #pragma omp simd aligned(source, destination: 64) simdlen(64)
    for (uint32_t i = 1; i < bytesToProcess; i++)
    {
        destination[i] = source[i] - source[i - 1];
    }
}

void Compressor::compressAdaptive()
{   
    int threadNumber = omp_get_thread_num();

    #pragma omp master
    {
        DEBUG_PRINT("Static adaptive compression started");
    }

    {
        analyzeImageAdaptive();
    }
    #pragma omp barrier
    
    #pragma omp master
    {
        for (uint32_t i = 0; i < _blockCount; i++)
        {
            AdaptiveTraversals bestTraversal = HORIZONTAL;
            int32_t bestRleCount = INT32_MIN;
            for (uint8_t j = 0; j < 2; j++)
            {
                if (_rlePerBlockCounts[j][i] > bestRleCount)
                {
                    bestRleCount = _rlePerBlockCounts[j][i];
                    bestTraversal = static_cast<AdaptiveTraversals>(j);
                }
                _bestBlockTraversals[i] = bestTraversal;
            }
        }

        #pragma omp task
        {
            buildHuffmanTree();
            populateCodeTable();
        }

        for (uint32_t i = 0; i < _blocksPerColumn; i++)
        {
            for (uint32_t j = 0; j < _blocksPerRow; j++)
            {
                switch (_bestBlockTraversals[i * _blocksPerRow + j])
                {
                case HORIZONTAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        serializeBlock(_sourceBuffer, _destinationBuffer, j, i);
                    }
                    break;
                
                case VERTICAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        transposeSerializeBlock(_sourceBuffer, _destinationBuffer, j, i);
                    }
                    break;
                }
            }
        }
    }
    #pragma omp taskwait
    #pragma omp barrier

    uint64_t startingIdx;
    {
        transformRLE(_destinationBuffer, reinterpret_cast<uint16_t *>(_sourceBuffer), _compressedSizes[threadNumber], startingIdx);
    }
    #pragma omp barrier

    #pragma omp single
    {
        // exclusive scan of compressed sizes
        _compressedSizesExScan[0] = 0;
        for (int32_t i = 0; i < _numberOfThreads; i++)
        {
            _compressedSizesExScan[i + 1] = _compressedSizesExScan[i] + _compressedSizes[i];
        }

        DEBUG_PRINT("Total compressed size: " << _compressedSizesExScan[_numberOfThreads]);
    }
    // implicit barrier

    // all threads pack the compressed data back into the initial buffer
    {
        copy(_sourceBuffer + startingIdx, _sourceBuffer + startingIdx + _compressedSizes[threadNumber], _destinationBuffer + _compressedSizesExScan[threadNumber]);
    }
    #pragma omp barrier

    #pragma omp master 
    {
        createHeader();
        
        DEBUG_PRINT("Static adaptive compression finished");
    }
}

void Compressor::compressStaticModel()
{
    {
        applyDiferenceModel(_sourceBuffer, _destinationBuffer);
    }
    #pragma omp barrier

    #pragma omp single
    {
        swap(_sourceBuffer, _destinationBuffer);
    }
    // implicit barrier

    compressStatic();
}

void Compressor::compressAdaptiveModel()
{
    {
        applyDiferenceModel(_sourceBuffer, _destinationBuffer);
    }
    #pragma omp barrier

    #pragma omp single
    {
        swap(_sourceBuffer, _destinationBuffer);
    }
    // implicit barrier

    compressAdaptive();
}


void Compressor::decomposeDataBetweenThreads(symbol_t *data, uint32_t &bytesPerThread, uint32_t &startingIdx, symbol_t &firstSymbol)
{
    int threadNumber = omp_get_thread_num();
    bytesPerThread = (_size + _numberOfThreads - 1) / _numberOfThreads;
    bytesPerThread += bytesPerThread & 0x1; // ensure even number of bytes per thread, TODO solve single thread case
    startingIdx = bytesPerThread * threadNumber;
    firstSymbol = data[startingIdx];
    if (threadNumber == _numberOfThreads - 1 && _numberOfThreads > 1) // last thread
    {
        // ensure last symbol in a block is different from the first in next block
        data[startingIdx] = ~data[startingIdx - 1];
        data[_size] = ~data[_size - 1];

        bytesPerThread -= bytesPerThread * _numberOfThreads - _size; // ensure last thread does not run out of bounds
    }
    else if (threadNumber != 0) // remaining threads apart from the first one
    {
        data[startingIdx] = ~data[startingIdx - 1];
    }
    else // single threaded execution
    {
        data[_size] = ~data[_size - 1];
    }
}

void Compressor::compress(string inputFileName, string outputFileName)
{
    clearMemory();
    readInputFile(inputFileName);
    
    if (_adaptive)
    {
        initializeBlockTypes();
    }
    
    #pragma omp parallel
    {
        DEBUG_PRINT("Thread " << omp_get_thread_num() << " started");
        if (_model)
        {
            if (_adaptive)
            {
                compressAdaptiveModel();
            }
            else
            {
                compressStaticModel();
            }
        }
        else
        {
            if (_adaptive)
            {
                compressAdaptive();
            }
            else
            {
                compressStatic();
            }
        }
        DEBUG_PRINT("Thread " << omp_get_thread_num() << " finished");
    }
    DEBUG_PRINT("All threads finished");

    writeOutputFile(outputFileName, inputFileName);
}
