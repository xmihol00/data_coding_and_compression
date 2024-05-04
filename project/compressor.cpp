/* =======================================================================================================================================================
 * Project:         Huffman RLE compression and decompression
 * Author:          David Mihola (xmihol00)
 * E-mail:          xmihol00@stud.fit.vutbr.cz
 * Date:            12. 5. 2024
 * Description:     A parallel implementation of a compression algorithm based on Huffman coding and RLE transformation. 
 *                  This file contains the implementation of the compressor.
 * ======================================================================================================================================================= */

#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint64_t width, int32_t numberOfThreads)
    : HuffmanRLECompression(model, adaptive, width, numberOfThreads) { }

void Compressor::freeMemory()
{
    if (_sourceBuffer != nullptr)
    {
        free(_sourceBuffer);
    }

    if (_destinationBuffer != nullptr)
    {
        free(_destinationBuffer);
    }

    if (_rlePerBlockCounts != nullptr)
    {
        delete[] _rlePerBlockCounts;
    }

    if (_blockTypes != nullptr)
    {
        delete[] _blockTypes;
    }

    if (_bestBlockTraversals != nullptr)
    {
        delete[] _bestBlockTraversals;
    }
}

void Compressor::readInputFile()
{
    DEBUG_PRINT("Reading input file");
    startReadInputFileTimer(); // NOP if partial measurements are disabled

    ifstream inputFile(_inputFileName, ios::binary);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file '" << _inputFileName << "'." << endl;
        exit(INPUT_FILE_ERROR);
    }

    // get the size of the input file
    inputFile.seekg(0, ios::end);
    _size = inputFile.tellg();
    captureUncompressedFileSize(); // NOP if performance measurements are disabled
    inputFile.seekg(0, ios::beg);

    if (_size == 0)
    {
        cerr << "Warning: Compressing an empty file." << endl;
        if (_width != 0)
        {
            cerr << "         Specified width does not match the file size." << endl;
        }

        ofstream outputFile(_outputFileName, ios::binary);
        if (!outputFile.is_open())
        {
            cerr << "Error: Unable to open output file '" << _outputFileName << "'." << endl;
            exit(OUTPUT_FILE_ERROR);
        }

        FirstByteHeader &header = reinterpret_cast<FirstByteHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.setNotCompressed();
        outputFile.write(reinterpret_cast<char *>(&_headerBuffer), sizeof(FirstByteHeader));
        outputFile.close();

        freeMemory();

        exit(SUCCESS);
    }

    if (_numberOfThreads > 1 && _size < (static_cast<uint64_t>(_numberOfThreads) << 4)) // size must be at least 16 times the number of threads
    {
        cerr << "Warning: Input file is too small for multi-threaded compression with " << _numberOfThreads << ".\n"; 
        do
        {
            _numberOfThreads >>= 1;
        }
        while (_numberOfThreads > 1 && _size < (static_cast<uint64_t>(_numberOfThreads) << 4));
        omp_set_num_threads(_numberOfThreads);
        cerr << "         Proceeding with " << _numberOfThreads << " threads instead." << endl;
    }

    _height = _size / _width;
    if (_height * _width != _size) // verify that file is rectangular
    {
        cerr << "Error: The input file size is not a multiple of the specified width." << endl;
        exit(FILE_SIZE_ERROR);
    }

    if (_adaptive && _height >= (1UL << MAX_BITS_PER_FILE_DIMENSION)) // verify that he height is within the allowed range
    {
        cerr << "Error: Input file is too large, height out of range, maximum allowed height is " 
             << (1UL << MAX_BITS_PER_FILE_DIMENSION) - 1 << " (2^" << MAX_BITS_PER_FILE_DIMENSION << "-1)." << endl;
        exit(FILE_SIZE_ERROR);
    }
    else if (_adaptive && _width >= (1UL << MAX_BITS_PER_FILE_DIMENSION)) // verify that the width is within the allowed range
    {
        cerr << "Error: Input file is too large, width out of range, maximum allowed width is " 
             << (1UL << MAX_BITS_PER_FILE_DIMENSION) - 1 << " (2^" << MAX_BITS_PER_FILE_DIMENSION << "-1)." << endl;
        exit(FILE_SIZE_ERROR);
    }
    else if (!_adaptive && _size >= (1UL << MAX_BITS_FOR_FILE_SIZE)) // verify that the file size is within the allowed range
    {
        cerr << "Error: Input file is too large, maximum allowed size is " 
             << (1UL << MAX_BITS_FOR_FILE_SIZE) - 1 << " (2^" << MAX_BITS_FOR_FILE_SIZE << "-1)." << endl;
        exit(FILE_SIZE_ERROR);
    }

    if ((_width % BLOCK_SIZE || _height % BLOCK_SIZE) && _adaptive) // verify that file is divisible by block size
    {
        cerr << "Warning: The input file width and height is not divisible by a built in adaptive granularity of " << BLOCK_SIZE << "." << endl;
        cerr << "         Static compression will be used instead." << endl;
        _adaptive = false; // disable adaptive compression 
    }

    uint64_t roundedSize = ((_size + _numberOfThreads - 1) / _numberOfThreads) * _numberOfThreads + 1;
    _threadPadding = roundedSize >> 4; // ensure there is some spare space if compressed block is larger than the original block
    _threadPadding += _threadPadding & 0b1; // ensure the padding is even
    roundedSize += _threadPadding * _numberOfThreads;

    // these buffers will be used in a ping-pong fashion
    _sourceBuffer = static_cast<symbol_t *>(aligned_alloc(64, roundedSize * sizeof(symbol_t)));
    _destinationBuffer = static_cast<symbol_t *>(aligned_alloc(64, roundedSize * sizeof(symbol_t)));
    if (_sourceBuffer == nullptr || _destinationBuffer == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the input file." << endl;
        exit(MEMORY_ALLOCATION_ERROR);
    }

    inputFile.read(reinterpret_cast<char *>(_sourceBuffer), _size);
    inputFile.close();

    stopReadInputFileTimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Input file read");
}

void Compressor::computeHistogram()
{
    DEBUG_PRINT("Thread " << omp_get_thread_num() << ": computing histogram");
    startHistogramComputationTimer(); // NOP if partial measurements are disabled

    int threadId = omp_get_thread_num();
    uint64_t *intHistogram = _intHistogram;
    symbol_t *buffer = _sourceBuffer;
    uint16_t numberOfThreads = _numberOfThreads > MAX_HISTOGRAM_THREADS ? MAX_HISTOGRAM_THREADS : _numberOfThreads;

    if (threadId < MAX_HISTOGRAM_THREADS)
    {
        uint64_t symbolsToProcess = (_size + numberOfThreads - 1) / numberOfThreads;
        intHistogram += threadId * NUMBER_OF_SYMBOLS;
        buffer += threadId * symbolsToProcess;
        if (threadId == numberOfThreads - 1)
        {
            symbolsToProcess = _size - threadId * symbolsToProcess;
        }

        #pragma omp simd aligned(intHistogram: 64) simdlen(8)
        for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            intHistogram[i] = 0;
        }

        uint8_t lastSymbol = ~buffer[0]; // ensure the first symbol at first iteration is different
        uint64_t sameSymbolCount = 0;
        bool same;
        for (uint64_t i = 0; i < symbolsToProcess; i++)
        {
            same = buffer[i] == lastSymbol;
            lastSymbol = buffer[i];
            sameSymbolCount *= same; // reset the counter if the symbol is different
            sameSymbolCount++;
            intHistogram[buffer[i]] += sameSymbolCount <= SAME_SYMBOLS_TO_REPETITION; // count only the first 3 occurrences of the same symbol
        }
    }
    #pragma omp barrier // wait for all threads to finish the histogram computation

    // it is necessary to have 2 if statements, because the barrier must be called by all threads, even by those that do not compute the histogram
    if (threadId < MAX_HISTOGRAM_THREADS)
    {
        uint16_t symbolsPerThread = NUMBER_OF_SYMBOLS / numberOfThreads;
        uint16_t threadOffset = threadId * symbolsPerThread;

        // sum up the histograms computed by individual threads
        intHistogram = _intHistogram + threadOffset;
        for (uint8_t i = 1; i < numberOfThreads; i++)
        {
            #pragma omp simd aligned(intHistogram: 64) simdlen(8)
            for (uint16_t j = 0; j < symbolsPerThread; j++) // each thread sums up part of the histogram
            {
                intHistogram[j] += intHistogram[i * NUMBER_OF_SYMBOLS + j];
            }
        }

        uint32_t *structHistogram = reinterpret_cast<uint32_t *>(_structHistogram) + threadOffset;
        #pragma omp simd aligned(structHistogram, intHistogram: 64) simdlen(16)
        for (uint32_t i = 0; i < symbolsPerThread; i++) // each thread stores the frequency and the symbol in different part of the histogram
        {
            // clamp the frequency to the maximum value
            intHistogram[i] = intHistogram[i] >= (1UL << MAX_BITS_FOR_FREQUENCY) ? (1UL << MAX_BITS_FOR_FREQUENCY) - 2 : intHistogram[i];
            structHistogram[i] = intHistogram[i] << 8; // store the frequency in the struct at the 24 MSBs
            structHistogram[i] |= i + threadOffset;    // store the symbol index in the struct at the 8 LSBs
            structHistogram[i] = intHistogram[i] == 0 ? UINT32_MAX : structHistogram[i];
        }
    }
    #pragma omp barrier // wait for all threads to construct the frequency-symbol pairs

    stopHistogramComputationTimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Thread " << threadId << ": histogram computed");
}

void Compressor::buildHuffmanTree()
{
    DEBUG_PRINT("Building Huffman tree");
    startHuffmanTreeBuildTimer(); // NOP if partial measurements are disabled

    uint32_t *symbolsParentsDepths = reinterpret_cast<uint32_t *>(_symbolsParentsDepths);
    uint16_t *parentsSortedIndices = _parentsSortedIndices;
    uint16_t *symbolsPerDepth = _symbolsPerDepth;

    // clear the arrays
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
    #pragma omp simd aligned(symbolsPerDepth: 64) simdlen(16)
    for (uint16_t i = 0; i < MAX_NUMBER_OF_CODES * 2; i++)
    {
        symbolsPerDepth[i] = 0;
    }
    
    uint16_t sortedIdx = 0;
    union Minimum
    {
        uint32_t intMin;
        FrequencySymbolIndex structMin;
    };
    Minimum firstMin;
    Minimum secondMin;

    while (true)
    {
    #if __AVX512F__
        uint32v16_t min1 = _vectorHistogram[0];
        uint32v16_t min2 = _vectorHistogram[1];
        for (uint8_t i = 2; i < NUMBER_OF_SYMBOLS / 16 ; i += 2)
        {
            uint32v16_t current1 = _vectorHistogram[i];
            uint32v16_t current2 = _vectorHistogram[i + 1];
            
            min1 = _mm512_min_epu32(min1, current1);
            min2 = _mm512_min_epu32(min2, current2);
        }
        min1 = _mm512_min_epu32(min1, min2);
        firstMin.intMin = _mm512_reduce_min_epu32(min1);
        _structHistogram[firstMin.structMin.index] = EMPTY_FREQUENCY_SYMBOL_INDEX;
        
        min1 = _vectorHistogram[0];
        min2 = _vectorHistogram[1];
        for (uint8_t i = 2; i < NUMBER_OF_SYMBOLS / 16 ; i += 2)
        {
            uint32v16_t current1 = _vectorHistogram[i];
            uint32v16_t current2 = _vectorHistogram[i + 1];
            min1 = _mm512_min_epu32(min1, current1);
            min2 = _mm512_min_epu32(min2, current2);
        }
        min1 = _mm512_min_epu32(min1, min2);
        
        secondMin.intMin = _mm512_reduce_min_epu32(min1);
    #else
        firstMin.intMin = min_element(reinterpret_cast<uint32_t *>(_structHistogram), reinterpret_cast<uint32_t *>(_structHistogram) + NUMBER_OF_SYMBOLS)[0];
        _structHistogram[firstMin.structMin.index] = EMPTY_FREQUENCY_SYMBOL_INDEX;
        secondMin.intMin = min_element(reinterpret_cast<uint32_t *>(_structHistogram), reinterpret_cast<uint32_t *>(_structHistogram) + NUMBER_OF_SYMBOLS)[0];
    #endif

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

        if (secondMin.intMin == UINT32_MAX)
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
        uint64_t nextMin = firstMin.intMin + secondMin.intMin;
        nextNode.intMin = nextMin >= UINT32_MAX ? MAX_VALID_FREQUENCY_SYMBOL_INDEX : nextMin;
        nextNode.structMin.index = secondIndex;
        _structHistogram[secondIndex] = nextNode.structMin;
        _parentsSortedIndices[secondIndex] = sortedIdx;
    }

    int16_t symbolsDepthsIdx = 0;
    for (uint16_t i = sortedIdx; i > 0; i--)
    {
        uint8_t nextDepth = _symbolsParentsDepths[_symbolsParentsDepths[i].parent].depth + 1;
        if (_symbolsParentsDepths[i].depth)
        {
            _symbols[symbolsDepthsIdx] = _symbolsParentsDepths[i].symbol;
            _depths[symbolsDepthsIdx] = nextDepth;
            symbolsPerDepth[nextDepth]++;
            symbolsDepthsIdx++;
        }
        _symbolsParentsDepths[i].depth = nextDepth;
    }
    _numberOfSymbols = symbolsDepthsIdx;

    if (_numberOfSymbols == 1)
    {
        _adjustedDepths[0] = 1;
        DEBUG_PRINT("Huffman tree built with only one symbol");
        return;
    }

#if __AVX512BW__ && __AVX512VL__
    // tree balancing to reduce the depth to maximum of 16 and the number of prefixes to maximum of 32
    uint16_t lastCode = -1;
    uint8_t lastDepth = 0;
    uint16_t numberOfPrefixes = MAX_NUMBER_OF_PREFIXES + 1;
    uint16_t addedSymbols = 0;
    uint8_t roundingShift = 0;      // increased by 1 after each iteration
    int16_t processedSymbols = 0;
    uint8_t maxDepth = _depths[symbolsDepthsIdx - 1];
    uint8_t adjustedDepthIdx = 0;
    while (numberOfPrefixes > MAX_NUMBER_OF_PREFIXES || maxDepth > MAX_CODE_LENGTH)
    {
        // reset the state
        numberOfPrefixes = 0;
        addedSymbols = 0;
        lastCode = -1;
        lastDepth = 0;
        processedSymbols = _numberOfSymbols;
        adjustedDepthIdx = 0;

        // build a new huffman tree
        for (uint16_t i = 0; i < MAX_NUMBER_OF_CODES * 2 && processedSymbols; i++)
        {
            uint16_t adjustedSymbols = symbolsPerDepth[i];
            
            if (symbolsPerDepth[i] != 0 || addedSymbols >= processedSymbols)
            {
                adjustedSymbols += addedSymbols;
                adjustedSymbols = processedSymbols < adjustedSymbols ? processedSymbols : adjustedSymbols;
                lastCode = (lastCode + 1) << (i - lastDepth); // get the next canonical Huffman code
                lastDepth = i;
                uint16_t symbolsToInsert = adjustedSymbols;
                if (symbolsToInsert != processedSymbols)
                {
                    // remove the last 'roundingShift' least significant bits
                    symbolsToInsert >>= roundingShift;
                    symbolsToInsert <<= roundingShift;
                }
                addedSymbols = adjustedSymbols - symbolsToInsert; // save the symbols that were excluded by the rounding
                
                while (symbolsToInsert)
                {
                    uint8_t trailingZeros = countr_zero(lastCode);
                    uint32_t fits = 1 << trailingZeros;
                    fits = symbolsToInsert < fits ? symbolsToInsert : fits;
                    lastCode += fits;
                    symbolsToInsert -= fits;
                    processedSymbols -= fits;
                    numberOfPrefixes++;

                    for (uint16_t j = 0; j < fits; j++)
                    {
                        _adjustedDepths[adjustedDepthIdx++] = i;
                    }
                }
                lastCode--;
            }
            // multiply the number of symbols that were excluded at the current depth by 2, meaning that this many additional symbols can be added at the next depth
            addedSymbols <<= 1;
            maxDepth = i;
        }
        roundingShift++;
    }
#else
    // reduce the depth to maximum of 15 (one less to adjust for possible overflow in the next step)
    _depths[symbolsDepthsIdx] = 0; // ensure that the memory after the last depth is initialized
    symbolsDepthsIdx--;
    uint8_t maxDepth = _depths[symbolsDepthsIdx];
    constexpr uint16_t decreasedMaxCodeLength = MAX_CODE_LENGTH - 1;
    if (maxDepth > decreasedMaxCodeLength)
    {
        int16_t depthIdx = symbolsDepthsIdx;
        uint8_t maxDepthShift = maxDepth - decreasedMaxCodeLength;
        int32_t debt = 0; // the debt corresponds to the number of symbols that need to be moved to the next depth
        while (_depths[depthIdx] > decreasedMaxCodeLength)
        {
            uint8_t maxDepthDifference = _depths[depthIdx] - decreasedMaxCodeLength;
            debt += ((1 << (maxDepthDifference)) - 1) << (maxDepthShift - maxDepthDifference); // deeper symbols are more expensive to move
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
            debt -= 1 << (maxDepth - _depths[depthIdx] - 1); // symbols higher in the tree repay more debt when moved lower
            _depths[depthIdx]++; // move symbol deeper 
            depthIdx--;
        }     
    }
    
    // rebalance the tree, set the number of symbols at each depth to a power of 2 and increasing with the depth
    uint16_t depthIdx = 0;
    uint16_t nodesToRebalance = 1 << (_depths[0] - 1); // number of nodes that fits at a given depth
    uint8_t depth = _depths[0];
    while (depthIdx < _numberOfSymbols)
    {
        // minimum between remaining nodes to be balanced and the number of nodes that fits at the current depth
        nodesToRebalance = depthIdx + nodesToRebalance < _numberOfSymbols ? nodesToRebalance : _numberOfSymbols - depthIdx;
        for (uint16_t i = 0; i < nodesToRebalance; i++)
        {
            _depths[depthIdx++] = depth; // set the depth of the symbols
        }

        if (depth < _depths[depthIdx]) // fist yet unbalanced symbols is at a deeper depth than the last balanced one
        {
            nodesToRebalance <<= _depths[depthIdx] - depth; // increase the number of nodes that fits at the next depth
            depth = _depths[depthIdx];
        }
        else if (depth == _depths[depthIdx]) // the depth of unbalanced and balanced nodes is the same, increase the depth of the unbalanced nodes
        {
            nodesToRebalance <<= 1;
            depth += 1;
        }

        depth++;
    }
    _adjustedDepths = _depths;
#endif

    stopHuffmanTreeBuildTimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Huffman tree built");
}

void Compressor::populateCodeTable()
{
    DEBUG_PRINT("Populating code table");
    startCodeTablePopulationTimer(); // NOP if partial measurements are disabled

    uint32_t *codeTable = reinterpret_cast<uint32_t *>(_codeTable);
    uint32_t *symbolsAtDepths = reinterpret_cast<uint32_t *>(_symbolsAtDepths);

    // clear the arrays
    #pragma omp simd aligned(codeTable: 64) simdlen(16)
    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        codeTable[i] = 0;
    }
    #pragma omp simd aligned(symbolsAtDepths: 64) simdlen(16)
    for (uint16_t i = 0; i < MAX_NUMBER_OF_CODES * 8; i++)
    {
        symbolsAtDepths[i] = 0;
    }

    // sort symbols at depths using bitmaps, each depth has a 256-bit bitmap, where each bit represents a symbol 
    // (correspondng set bit means the symbol is at the depth)
    _usedDepths = 0;
    for (uint16_t i = 0; i < _numberOfSymbols; i++)
    {
        uint64_t bits[4];
        // clear the bitmap
    #if __AVX2__
        reinterpret_cast<uint64v4_t *>(bits)[0] = _mm256_setzero_si256();
    #else
        bits[0] = 0;
        bits[1] = 0;
        bits[2] = 0;
        bits[3] = 0;
    #endif

        // place the symbol to a correct 64-bit part of the bitmap
        _usedDepths |= 1U << _adjustedDepths[i];
        uint8_t symbol = 255 - _symbols[i];
        bits[3] = (uint64_t)(symbol < 64) << symbol;
        bits[2] = (uint64_t)(symbol < 128 && symbol >= 64) << (symbol - 64);
        bits[1] = (uint64_t)(symbol < 192 && symbol >= 128) << (symbol - 128);
        bits[0] = (uint64_t)(symbol >= 192) << (symbol - 192);

        // or the bitmap of the current symbol with the bitmap of all the other symbols at the current depth
    #if __AVX2__
        _symbolsAtDepths[_adjustedDepths[i]] = _mm256_or_si256(_symbolsAtDepths[_adjustedDepths[i]], reinterpret_cast<uint64v4_t *>(bits)[0]);
    #else
        reinterpret_cast<uint64_t *>(_symbolsAtDepths + _adjustedDepths[i])[0] |= bits[0];
        reinterpret_cast<uint64_t *>(_symbolsAtDepths + _adjustedDepths[i])[1] |= bits[1];
        reinterpret_cast<uint64_t *>(_symbolsAtDepths + _adjustedDepths[i])[2] |= bits[2];
        reinterpret_cast<uint64_t *>(_symbolsAtDepths + _adjustedDepths[i])[3] |= bits[3];
    #endif
    }

    uint16_t lastCode = -1;
    uint8_t depthIndex = 0;
    uint8_t delta = 0;
    _maxSymbolsPerDepth = 0;
    _mostPopulatedDepthIdx = 0;

    // populate the code table
    for (uint8_t i = 0; i < MAX_NUMBER_OF_CODES; i++)
    {
        if (_usedDepths & (1UL << i))
        {
            _symbolsAtDepths[depthIndex++] = _symbolsAtDepths[i];
        #if __AVX2__
            uint64v4_t bitsVector = _mm256_load_si256(_symbolsAtDepths + i);
            uint64_t *bits = reinterpret_cast<uint64_t *>(&bitsVector);
        #else
            uint64_t bits[4];
            bits[0] = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i)[0];
            bits[1] = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i)[1];
            bits[2] = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i)[2];
            bits[3] = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i)[3];
        #endif
            lastCode = (lastCode + 1) << delta; // get the next canonical Huffman code for the 1st symbol at a given depth
            lastCode--; // move one code back for easier processing in the following loop
            uint16_t oldCode = lastCode;

            for (uint8_t j = 0; j < 4; j++) // for the 4 64-bit parts of the bitmap
            {
                uint8_t symbol = j << 6; // offset of the current 64-bit part of the bitmap in the 256-bit bitmap
                uint8_t leadingZeros;
                while ((leadingZeros = countl_zero(bits[j])) < 64) // until the 64-bit part of the bitmap is not empty
                {
                    uint8_t adjustedSymbol = symbol + leadingZeros; // value of the symbol
                    lastCode++;
                    bits[j] ^= 1UL << (63 - leadingZeros); // remove the current symbol from the bitmap
                    _codeTable[adjustedSymbol].code = lastCode;
                    _codeTable[adjustedSymbol].length = i; // length of the symbols is equal to its depth
                }
            }
            delta = 0;

            uint16_t codeCount = lastCode - oldCode; // number of symbols added at the current depth
            _maxSymbolsPerDepth = max(_maxSymbolsPerDepth, codeCount);
            if (_maxSymbolsPerDepth == codeCount) // continuously update the most populated depth
            {
                _mostPopulatedDepthIdx = depthIndex - 1;
                _mostPopulatedDepth = i;
            }
        }
        delta++;
    }

    stopCodeTablePopulationTimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Code table populated");
}

void Compressor::transformRLE(symbol_t *sourceData, uint16_t *compressedData, uint64_t &compressedSize, uint64_t &startingIdx)
{
    int threadNumber = omp_get_thread_num();
    DEBUG_PRINT("Thread " << threadNumber << ": transforming RLE");
    startTransformRLETimer(); // NOP if partial measurements are disabled

    uint64_t bytesToCompress = (_size + _numberOfThreads - 1) / _numberOfThreads;
    bytesToCompress += bytesToCompress & 0b1; // ensure the number of bytes to compress is even
    startingIdx = bytesToCompress * threadNumber;
    symbol_t firstSymbol = sourceData[startingIdx];
    #pragma omp barrier // wait for all threads to get the first symbol

    // ensure last thread does not run out of bounds
    bytesToCompress = (threadNumber == _numberOfThreads - 1) ? _size - bytesToCompress * (_numberOfThreads - 1) : bytesToCompress; 
    // ensure last symbol in a block is different from the first in next block
    sourceData[startingIdx + bytesToCompress] = ~sourceData[startingIdx + bytesToCompress - 1];

    sourceData += startingIdx;
    startingIdx += _threadPadding * threadNumber;
    compressedData += startingIdx >> 1;

    compressedData[0] = 0;
    uint64_t currentCompressedIdx = 0;
    uint64_t nextCompressedIdx = 1;
    uint64_t sourceDataIdx = 1;
    uint64_t sameSymbolCount = 1;
    symbol_t current = firstSymbol;
    uint8_t chunkIdx = 0;

    uint64_t shortsToCompress = (bytesToCompress + _threadPadding - 2) >> 1;
    // until there is something to compress or space for the compressed is not exhausted, in the latter case no compression is achieved
    while (sourceDataIdx <= bytesToCompress && nextCompressedIdx < shortsToCompress)
    {
        if (sourceData[sourceDataIdx] == current) // repeating symbol
        {
            sameSymbolCount++;
        }
        else
        {
            // retrieve the corresponding Huffman code and its length for the current symbol
            uint16_t codeLength = _codeTable[current].length;
            uint16_t code = _codeTable[current].code << (16 - codeLength);

            // store the Huffman code in the compressed data up to 3 times
            for (uint64_t i = 0; i < sameSymbolCount && i < SAME_SYMBOLS_TO_REPETITION; i++)
            {
                // separate the code to two parts, that are stored in two 16-bit parts of the compressed data
                uint16_t downShiftedCode = code >> chunkIdx;
                uint16_t upShiftedCode = code << (16 - chunkIdx);

                // store the two parts of the code in the compressed data
                compressedData[currentCompressedIdx] |= downShiftedCode;
                compressedData[nextCompressedIdx] = 0; // clear the next 16-bit part first
                compressedData[nextCompressedIdx] |= upShiftedCode;

                chunkIdx += codeLength;
                bool moveChunk = chunkIdx >= 16;   // move to the next 16-bit part of the compressed data
                chunkIdx &= 15;                    // modulo 16
                currentCompressedIdx += moveChunk; // the next 16-bit part of the compressed data becomes the current part
                nextCompressedIdx += moveChunk;    // new next 16-bit part of the compressed data will be used in the next iteration
            }

            bool repeating = sameSymbolCount >= 3;
            sameSymbolCount -= 3; // the first 3 symbols were already compressed
            while (repeating)
            {
                // mask the lower bits of the number of repetitions and retrieve a partial number of repetitions
                uint16_t count = sameSymbolCount & ((~0U) >> (16 - BITS_PER_REPETITION_NUMBER));
                // decrease the number of repetitions by the number of repetitions already stored in 'count'
                sameSymbolCount >>= (BITS_PER_REPETITION_NUMBER - 1);
                repeating = sameSymbolCount > 0;                        // check if there are more to be stored
                count |= repeating << (BITS_PER_REPETITION_NUMBER - 1); // indicate to the decompressor that there are/aren't more repetitions
                count <<= (16 - BITS_PER_REPETITION_NUMBER);            // align the count to MSB

                // perform the same operation as for the Huffman code
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

            sameSymbolCount = 1; // reset the same symbol counter
            current = sourceData[sourceDataIdx];
        }
        sourceDataIdx++;
    }

    _compressionUnsuccessful = sourceDataIdx <= bytesToCompress; // there is still something to compress
    compressedSize = nextCompressedIdx * 2; // size in bytes

    stopTransformRLETimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Thread " << threadNumber << ": RLE transformed, compressed size: " << compressedSize);
}

void Compressor::createHeader()
{
    DEBUG_PRINT("Creating header");

    compressDepthMaps(); // compress the depth maps, that is the Huffman tree

    #if __AVX2__
        reinterpret_cast<uint16v16_t *>(_headerBuffer)[0] = _mm256_setzero_si256(); // clear the buffer
    #else
        reinterpret_cast<uint64_t *>(_headerBuffer)[0] = 0;
        reinterpret_cast<uint64_t *>(_headerBuffer)[1] = 0;
        reinterpret_cast<uint64_t *>(_headerBuffer)[2] = 0;
        reinterpret_cast<uint64_t *>(_headerBuffer)[3] = 0;
    #endif

    if (_numberOfThreads > 1) // multi-threaded compression
    {
        uint16_t maxBitsCompressedSizes;
        packCompressedSizes(maxBitsCompressedSizes);
    
        DepthBitmapsMultiThreadedHeader &header = reinterpret_cast<DepthBitmapsMultiThreadedHeader &>(_headerBuffer);
        // compose the header
        header.clearFirstByte();
        header.insertHeaderType(MULTI_THREADED);
        header.setVersion(0);
        header.setNumberOfThreads(_numberOfThreads);
        // retrieve the number of bits needed to store block sizes of compressed data by each thread
        header.setBitsPerBlockSize(maxBitsCompressedSizes);
        header.setCodeDepths(_usedDepths); // bitmap of used depths in the Huffman tree
        _headerSize = sizeof(DepthBitmapsMultiThreadedHeader);
        _threadBlocksSizesSize = (_numberOfThreads * maxBitsCompressedSizes + 7) / 8; // round to bytes
    }
    else // single-threaded compression
    {
        DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.insertHeaderType(SINGLE_THREADED);
        header.setVersion(0);
        header.setCodeDepths(_usedDepths);
        _headerSize = sizeof(DepthBitmapsHeader);
        _threadBlocksSizesSize = 0;
    }

    BasicHeader &header = reinterpret_cast<BasicHeader &>(_headerBuffer);
    if (_adaptive) // adaptive compression
    {
        header.insertHeaderType(ADAPTIVE);
        // use 2D description of the size of the compressed data
        header.setWidth(_width);
        header.setHeight(_height);

        _blockTypesByteSize = (_numberOfTraversalBlocks * BITS_PER_BLOCK_TYPE + 7) / 8; // round to bytes
        _blockTypes = new uint8_t[_blockTypesByteSize];
        for (uint32_t i = 0, j = 0; i < _blockTypesByteSize; i++, j += 4)
        {
            // pack the block types into bytes, 2 bits per block type
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
        header.setSize(_size); // use 1D description of the size of the compressed data
    }
    
    if (_model) // model was applied to the data before compression
    {
        header.insertHeaderType(MODEL);
    }
    else
    {
        header.insertHeaderType(DIRECT);
    }
    
    DEBUG_PRINT("Header created");
}

void Compressor::packCompressedSizes(uint16_t &maxBitsCompressedSizes)
{
    DEBUG_PRINT("Packing compressed sizes");

    // get the maximum number of bits needed to store the size of the compressed data by each thread
    uint32_t max = max_element(_compressedSizes, _compressedSizes + _numberOfThreads)[0];
    maxBitsCompressedSizes = 32 - countl_zero(max);

    uint8_t *packedCompressedSizes = reinterpret_cast<uint8_t *>(_compressedSizes); // store the packed sizes in the same buffer
    uint16_t idx = 0;
    uint8_t chunkIdx = 0;
    uint32_t compressedSize = _compressedSizes[0];
    _compressedSizes[0] = 0; // clear the first size
    for (uint8_t i = 0; i < _numberOfThreads; i++)
    {
        uint8_t bits = maxBitsCompressedSizes;
        compressedSize <<= 32 - maxBitsCompressedSizes; // align the size to MSB

        do // distribute the size to the 8-bit parts of the buffer
        {
            packedCompressedSizes[idx] |= compressedSize >> (24 + chunkIdx); // store the current part of the size
            uint8_t storedBits = bits < 8 - chunkIdx ? bits : 8 - chunkIdx;  // number of bits stored in the current byte
            compressedSize <<= storedBits;
            bits -= storedBits;     // decrease the number of bits left to be stored
            chunkIdx += storedBits;
            if (chunkIdx >= 8) // move to the next byte
            {
                chunkIdx &= 0x7;
                packedCompressedSizes[++idx] = 0; // clear the next byte
            }
        } 
        while (bits);
        compressedSize = _compressedSizes[i + 1];
    }
    
    DEBUG_PRINT("Compressed sizes packed");
}

void Compressor::compressDepthMaps()
{
    DEBUG_PRINT("Compressing depth maps");

    uint16_t numberOfMaps = popcount(_usedDepths);
    // put the unseen symbols at the 0th depth, which is going to be the last one (at the last index in the compressed depth maps array)
    if (_numberOfSymbols < 256)
    {
        // the unseen symbols can be obtained by XORing all the seen symbols with fully populated 256-bit bitmap
    #if __AVX2__
        _symbolsAtDepths[numberOfMaps] = _mm256_set1_epi32(0xffff'ffff);
        for (uint16_t i = 0; i < numberOfMaps; i++)
        {
            _symbolsAtDepths[numberOfMaps] = _mm256_xor_si256(_symbolsAtDepths[numberOfMaps], _symbolsAtDepths[i]);
        }
    #else
        uint64_t *symbolsUnpacked = reinterpret_cast<uint64_t *>(_symbolsAtDepths + numberOfMaps);
        for (uint16_t i = 0; i < 4; i++)
        {
            symbolsUnpacked[i] = 0xffff'ffff'ffff'ffff;
        }
        for (uint16_t i = 0; i < numberOfMaps; i++)
        {
            uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
            for (uint16_t j = 0; j < 4; j++)
            {
                symbolsUnpacked[j] ^= bits[j];
            }
        }
    #endif

        uint16_t numberOfUnseenSymbols = 256 - _numberOfSymbols;
        // update the depth with maximum number of symbols if there are more unseen symbols than seen at any depth
        if (numberOfUnseenSymbols > _maxSymbolsPerDepth)
        {
            _mostPopulatedDepth = 0;
            _mostPopulatedDepthIdx = numberOfMaps;
        }

        // mark the 0th depth as used
        _usedDepths |= 0b1;
        numberOfMaps++;
    }

    // first byte indicates the most populated depth, which is not compressed (it can be retrieved by XORing all the other depths)
    _compressedDepthMaps[0] = _mostPopulatedDepth;
    uint16_t compressedIdx = 1;

    for (uint16_t i = 0; i < numberOfMaps; i++)
    {
        if (i != _mostPopulatedDepthIdx)
        {   
            // use 32-bit mask to indicate 0 bytes in the 256-bit bitmap and store only the non-zero bytes
            uint8_t *depthBytes = reinterpret_cast<uint8_t *>(_symbolsAtDepths + i);
        #if __AVX512BW__ && __AVX512VL__
            uint32_t mask = _mm256_cmp_epi8_mask(_symbolsAtDepths[i], _mm256_setzero_si256(), _MM_CMPINT_NE);
        #else
            uint32_t mask = 0;
            for (uint8_t j = 0; j < 32; j++)
            {
                mask |= (depthBytes[j] != 0) << j;
            }
        #endif
            _compressedDepthMaps[compressedIdx++] = mask >> 24;
            _compressedDepthMaps[compressedIdx++] = mask >> 16;
            _compressedDepthMaps[compressedIdx++] = mask >> 8;
            _compressedDepthMaps[compressedIdx++] = mask;

            while (mask) // there are non-zero bytes in the bitmap
            {
                uint8_t firstSetBit = 31 - countl_zero(mask);
                mask ^= 1 << firstSetBit; // mark the non-zero byte as stored
                _compressedDepthMaps[compressedIdx++] = depthBytes[firstSetBit]; // store the non-zero byte
            }
        }
    }
    _compressedDepthMapsSize = compressedIdx; // store the size of the compressed depth maps in bytes

    DEBUG_PRINT("Depth maps compressed");
}

void Compressor::writeOutputFile()
{
    DEBUG_PRINT("Writing output file");
    startWriteOutputFileTimer(); // NOP if partial measurements are disabled

    ofstream outputFile(_outputFileName, ios::binary);
    if (!outputFile.is_open())
    {
        cerr << "Error: Unable to open output file '" << _outputFileName << "'." << endl;
        exit(OUTPUT_FILE_ERROR);
    }
    
    if ((_size < _headerSize + _threadBlocksSizesSize + sizeof(uint64v4_t) * popcount(_usedDepths) + _compressedSizesExScan[_numberOfThreads]) ||
        _compressionUnsuccessful) // the compressed size is larger than the original size or compression was unsuccessful (some thread run out of space)
    {
        // use the first byte to store the information about the unsuccessful compression
        FirstByteHeader &header = reinterpret_cast<FirstByteHeader &>(_headerBuffer);
        header.clearFirstByte();
        header.setNotCompressed();
        outputFile.write(reinterpret_cast<char *>(&_headerBuffer), sizeof(FirstByteHeader));

        // copy the input file to the output file
        ifstream inputFile(_inputFileName, ios::binary);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << _inputFileName << "'." << endl;
            exit(INPUT_FILE_ERROR);
        }
        inputFile.read(reinterpret_cast<char *>(_sourceBuffer), _size);   
        inputFile.close();

        outputFile.write(reinterpret_cast<char *>(_sourceBuffer), _size);

        // reset the compressed sizes for later data analysis capture if active
        _headerSize = sizeof(FirstByteHeader);
        _threadBlocksSizesSize = 0;
        _compressedDepthMapsSize = 0;
        _blockTypesByteSize = 0;
        _compressedSizesExScan[_numberOfThreads] = _size;
    }
    else // data successfully compressed
    {
        // write the header
        outputFile.write(reinterpret_cast<char *>(_headerBuffer), _headerSize);
        // write the sizes of the compressed data by each thread (if multi-threaded compression)
        outputFile.write(reinterpret_cast<char *>(_compressedSizes), _threadBlocksSizesSize);
        // write the compressed depth maps
        outputFile.write(reinterpret_cast<char *>(_compressedDepthMaps), _compressedDepthMapsSize);
        // write the block types (if adaptive compression)
        outputFile.write(reinterpret_cast<char *>(_blockTypes), _blockTypesByteSize);
        // write the compressed data
        outputFile.write(reinterpret_cast<char *>(_destinationBuffer), _compressedSizesExScan[_numberOfThreads]);
    }
    outputFile.close();
    
    // NOPs if data analysis is disabled
    captureHeaderSize();
    captureHeaderSize();
    captureCompressedDepthMapsSize();
    captureBlockTypesByteSize();
    captureCompressedDataSize();
    captureWholeCompressedFileSize();

    stopWriteOutputFileTimer(); // NOP if partial measurements are disabled
    DEBUG_PRINT("Output file written");
}

void Compressor::compressStatic()
{
    int threadNumber = omp_get_thread_num();
    computeHistogram(); // compute the histogram of frequencies of symbols in the input data

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

    if (_numberOfThreads > 1) // multi-threaded compression
    {
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
    else // single-threaded compression
    {
        createHeader();
        DEBUG_PRINT("Static compression finished");
    }
}

void Compressor::analyzeImageAdaptive()
{
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " analyzing image");
    startAnalyzeImageAdaptiveTimer(); // NOP if performance measurements are disabled

    #pragma omp for schedule(static) // each iteration should take approximately the same time
    for (uint32_t i = 0; i < _blocksPerColumn; i++)
    {
        uint64_t blockRowIdx = i * _width * BLOCK_SIZE;
        for (uint32_t j = 0; j < _blocksPerRow; j++)
        {
            uint64_t blockFirstIdx = blockRowIdx + j * BLOCK_SIZE;

            int32_t horizontalSameSymbolCount = -3;
            int32_t verticalSameSymbolCount = -3;
            int32_t majorDiagonalSameSymbolCount = -3;
            int32_t minorDiagonalSameSymbolCount = -3;

            int16_t horizontalRleCount = 0;
            int16_t verticalRleCount = 0;
            int16_t majorDiagonalRleCount = 0;
            int16_t minorDiagonalRleCount = 0;

            uint16_t horizontalLastSymbol = -1;
            uint16_t verticalLastSymbol = -1;
            uint16_t majorDiagonalLastSymbol = -1;
            uint16_t minorDiagonalLastSymbol = -1;

            uint64_t horizontalValueIdx;
            uint64_t verticalValueIdx;
            uint64_t majorDiagonalValueIdx;
            uint64_t minorDiagonalValueIdx;

            bool horizontalSameSymbol;
            bool verticalSameSymbol;
            bool majorDiagonalSameSymbol;
            bool minorDiagonalSameSymbol;

            for (uint32_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
            {
                horizontalValueIdx = blockFirstIdx + HORIZONTAL_ZIG_ZAG_ROW_INDICES[k] * _width + HORIZONTAL_ZIG_ZAG_COL_INDICES[k];
                verticalValueIdx = blockFirstIdx + VERTICAL_ZIG_ZAG_ROW_INDICES[k] * _width + VERTICAL_ZIG_ZAG_COL_INDICES[k];
                majorDiagonalValueIdx = blockFirstIdx + MAJOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[k] * _width + MAJOR_DIAGONAL_ZIG_ZAG_COL_INDICES[k];
                minorDiagonalValueIdx = blockFirstIdx + MINOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[k] * _width + MINOR_DIAGONAL_ZIG_ZAG_COL_INDICES[k];

                horizontalSameSymbol = _sourceBuffer[horizontalValueIdx] == horizontalLastSymbol;
                verticalSameSymbol = _sourceBuffer[verticalValueIdx] == verticalLastSymbol;
                majorDiagonalSameSymbol = _sourceBuffer[majorDiagonalValueIdx] == majorDiagonalLastSymbol;
                minorDiagonalSameSymbol = _sourceBuffer[minorDiagonalValueIdx] == minorDiagonalLastSymbol;

                horizontalLastSymbol = _sourceBuffer[horizontalValueIdx];
                verticalLastSymbol = _sourceBuffer[verticalValueIdx];
                majorDiagonalLastSymbol = _sourceBuffer[majorDiagonalValueIdx];
                minorDiagonalLastSymbol = _sourceBuffer[minorDiagonalValueIdx];

                horizontalRleCount += !horizontalSameSymbol && horizontalSameSymbolCount >= -1 ? horizontalSameSymbolCount : 0;
                verticalRleCount += !verticalSameSymbol && verticalSameSymbolCount >= -1 ? verticalSameSymbolCount : 0;
                majorDiagonalRleCount += !majorDiagonalSameSymbol && majorDiagonalSameSymbolCount >= -1 ? majorDiagonalSameSymbolCount : 0;
                minorDiagonalRleCount += !minorDiagonalSameSymbol && minorDiagonalSameSymbolCount >= -1 ? minorDiagonalSameSymbolCount : 0;
                
                horizontalSameSymbolCount++;
                verticalSameSymbolCount++;
                majorDiagonalSameSymbolCount++;
                minorDiagonalSameSymbolCount++;

                horizontalSameSymbolCount = horizontalSameSymbol ? horizontalSameSymbolCount : -3;
                verticalSameSymbolCount = verticalSameSymbol ? verticalSameSymbolCount : -3;
                majorDiagonalSameSymbolCount = majorDiagonalSameSymbol ? majorDiagonalSameSymbolCount : -3;
                minorDiagonalSameSymbolCount = minorDiagonalSameSymbol ? minorDiagonalSameSymbolCount : -3;
            }

            uint64_t blockIdx = i * _blocksPerRow + j;
            AdaptiveTraversals horizontalVertical = horizontalRleCount >= verticalRleCount ? HORIZONTAL_ZIG_ZAG : VERTICAL_ZIG_ZAG;
            int16_t horizontalVerticalRleCount = horizontalRleCount >= verticalRleCount ? horizontalRleCount : verticalRleCount;
            AdaptiveTraversals diagonal = majorDiagonalRleCount >= minorDiagonalRleCount ? MAJOR_DIAGONAL_ZIG_ZAG : MINOR_DIAGONAL_ZIG_ZAG;
            int16_t diagonalRleCount = majorDiagonalRleCount >= minorDiagonalRleCount ? majorDiagonalRleCount : minorDiagonalRleCount;
            _bestBlockTraversals[blockIdx] = horizontalVerticalRleCount >= diagonalRleCount ? horizontalVertical : diagonal;
        }
    }
    // implicit barrier

    stopAnalyzeImageAdaptiveTimer(); // NOP if performance measurements are disabled
    countTraversalTypes();           // NOP if performance measurements are disabled
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " image analyzed");
}

void Compressor::applyDiferenceModel(symbol_t *source, symbol_t *destination)
{
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " applying difference model");
    startApplyDiferenceModelTimer(); // NOP if performance measurements are disabled

    int threadNumber = omp_get_thread_num();
    uint64_t bytesToProcess = (_size + _numberOfThreads - 1) / _numberOfThreads; // last thread can run out of bounds, the memory is padded
    bytesToProcess += bytesToProcess & 0b1; // ensure even number of bytes per thread
    uint64_t startingIdx = bytesToProcess * threadNumber;

    source += startingIdx;
    destination += startingIdx;
    destination[0] = source[0];

    #pragma omp simd simdlen(64)
    for (uint64_t i = 1; i < bytesToProcess; i++)
    {
        destination[i] = source[i] - source[i - 1]; // subtract the previous symbol from the current one
    }

    stopApplyDiferenceModelTimer(); // NOP if performance measurements are disabled
    DEBUG_PRINT("Thread: " << omp_get_thread_num() << " difference model applied");
}

void Compressor::compressAdaptive()
{   
    int threadNumber = omp_get_thread_num();
    analyzeImageAdaptive();

    startSerializeTraversalTimer(); // NOP if performance measurements are disabled
    // serialize the blocks into 1D memory
    #pragma omp for schedule(static) // each iteration should take approximately the same time
    for (uint32_t i = 0; i < _blocksPerColumn; i++)
    {
        for (uint32_t j = 0; j < _blocksPerRow; j++)
        {
            const uint8_t *rowIndices = ROW_INDICES_LOOKUP_TABLE[_bestBlockTraversals[i * _blocksPerRow + j]];
            const uint8_t *colIndices = COL_INDICES_LOOKUP_TABLE[_bestBlockTraversals[i * _blocksPerRow + j]];
            serializeBlock(_sourceBuffer, _destinationBuffer, i, j, rowIndices, colIndices);
        }
    }
    stopSerializeTraversalTimer(); // NOP if performance measurements are disabled

    computeHistogram(); // compute the histogram of frequencies of symbols in the adapted input data
    #pragma omp single
    {
        buildHuffmanTree();
        populateCodeTable();
    } 
    // implicit barrier

    uint64_t startingIdx;
    transformRLE(_destinationBuffer, reinterpret_cast<uint16_t *>(_sourceBuffer), _compressedSizes[threadNumber], startingIdx);
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

    if (_numberOfThreads > 1) // multi-threaded compression
    {
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
    else
    {
        swap(_sourceBuffer, _destinationBuffer);
        createHeader();
        DEBUG_PRINT("Static adaptive compression finished");
    }
}

void Compressor::compressStaticModel()
{
    // apply the difference model
    applyDiferenceModel(_sourceBuffer, _destinationBuffer);
    #pragma omp barrier

    #pragma omp single
    {
        swap(_sourceBuffer, _destinationBuffer);
    }
    // implicit barrier

    // then use the static compression as it would be used without the model
    compressStatic();
}

void Compressor::compressAdaptiveModel()
{
    // apply the difference model
    applyDiferenceModel(_sourceBuffer, _destinationBuffer);
    #pragma omp barrier

    #pragma omp single
    {
        swap(_sourceBuffer, _destinationBuffer);
    }
    // implicit barrier

    // then use the adaptive compression as it would be used without the model
    compressAdaptive();
}

void Compressor::compress(string inputFileName, string outputFileName)
{
    startFullExecutionTimer(); // NOP if full measurements are disabled

    _inputFileName = inputFileName;
    _outputFileName = outputFileName;
    readInputFile();
    
    if (_adaptive) // adaptive compression
    {
        // memory for traversal block types must be allocated
        initializeBlockTypes();
    }
    
    DEBUG_PRINT("Starting compression with " << _numberOfThreads << " threads");
    #pragma omp parallel // launch threads, i.e. create a parallel region
    {
        // select the compression method based on the passed parameters
        if (_model)
        {
            if (_adaptive)
            {
                startAdaptiveCompressionWithModelTimer(); // NOP if algorithm measurements are disabled
                compressAdaptiveModel();
                stopAdaptiveCompressionWithModelTimer();  // NOP if algorithm measurements are disabled
            }
            else
            {
                startStaticCompressionWithModelTimer(); // NOP if algorithm measurements are disabled
                compressStaticModel();
                stopStaticCompressionWithModelTimer();  // NOP if algorithm measurements are disabled
            }
        }
        else
        {
            if (_adaptive)
            {
                startAdaptiveCompressionTimer(); // NOP if algorithm measurements are disabled
                compressAdaptive();
                stopAdaptiveCompressionTimer();  // NOP if algorithm measurements are disabled
            }
            else
            {
                startStaticCompressionTimer(); // NOP if algorithm measurements are disabled
                compressStatic();
                stopStaticCompressionTimer();  // NOP if algorithm measurements are disabled
            }
        }
    }

    writeOutputFile();
    freeMemory(); // free the allocated memory so that the object does not have to be deleted and next compression can be performed

    stopFullExecutionTimer();   // NOP if full measurements are disabled
    printPerformanceCounters(); // not NOP if any measurements are active
}
