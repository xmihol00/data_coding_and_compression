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

void Compressor::computeHistogram()
{
    DEBUG_PRINT("Computing histogram");

    for (uint32_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        reinterpret_cast<uint64_t *>(_histogram)[i] = 0;
        _histogram[i].symbol = i;
    }

    for (uint64_t i = 0; i < _size; i++)
    {
        _histogram[_sourceBuffer[i]].count++;
    }

    DEBUG_PRINT("Histogram computed");
}

void Compressor::buildHuffmanTree()
{
    DEBUG_PRINT("Building Huffman tree");

    // create the Huffman tree starting from first character that occurred at least once
    uint16_t histogramIndex = 0;
    while (_histogram[histogramIndex].count == 0) // filter out characters that did not occur
    {
        histogramIndex++;
    }

    // pointers to the same memory
    Leaf *leafTree = reinterpret_cast<Leaf *>(_tree);
    Node *nodeTree = _tree;

    if (histogramIndex == NUMBER_OF_SYMBOLS - 1) // only a single type of symbol
    {
        leafTree[1] = _histogram[histogramIndex];
        _treeIndex = 1;
    }
    else
    {
        leafTree[1] = _histogram[histogramIndex++];
        leafTree[2] = _histogram[histogramIndex++];
        nodeTree[3].count = leafTree[1].count + leafTree[2].count;
        nodeTree[3].left = 1;
        nodeTree[3].right = 2;
        _treeIndex = 3;

        _sortedNodesHead = NUMBER_OF_SYMBOLS - 1;
        _sortedNodesTail = NUMBER_OF_SYMBOLS - 1;
        _sortedNodes[NUMBER_OF_SYMBOLS - 1] = nodeTree[3];
        _sortedNodes[NUMBER_OF_SYMBOLS - 1].left = 3;

        bool connectTree = false;

        while (histogramIndex < NUMBER_OF_SYMBOLS)
        {
            if (histogramIndex + 1 < NUMBER_OF_SYMBOLS && _histogram[histogramIndex + 1].count < _sortedNodes[_sortedNodesHead].count) // node from 2 symbols
            {
                leafTree[++_treeIndex] = _histogram[histogramIndex++];
                leafTree[++_treeIndex] = _histogram[histogramIndex++];

                ++_treeIndex;
                nodeTree[_treeIndex].count = leafTree[_treeIndex - 2].count + leafTree[_treeIndex - 1].count;
                nodeTree[_treeIndex].left = _treeIndex - 2;
                nodeTree[_treeIndex].right = _treeIndex - 1;

                connectTree = _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count;
            }
            else // node from 1 symbol and already existing sub-tree
            {
                leafTree[++_treeIndex] = _histogram[histogramIndex++];
                connectTree = true;
            }

            if (connectTree) // newly created node must be connected to the existing tree
            {
                do
                {
                    nodeTree[_treeIndex + 1].count = _sortedNodes[_sortedNodesHead].count + nodeTree[_treeIndex].count;
                    nodeTree[_treeIndex + 1].left = _sortedNodes[_sortedNodesHead++].left;
                    nodeTree[_treeIndex + 1].right = _treeIndex;
                    _treeIndex++;
                } while (_sortedNodesHead <= _sortedNodesTail && _sortedNodes[_sortedNodesHead].count <= nodeTree[_treeIndex].count);

                _sortedNodes[++_sortedNodesTail] = nodeTree[_treeIndex];
                _sortedNodes[_sortedNodesTail].left = _treeIndex;

                // buble sort the new node into the sorted nodes
                uint16_t idx = _sortedNodesTail - 1;
                Node tail = _sortedNodes[_sortedNodesTail];
                while (idx >= _sortedNodesHead && _sortedNodes[idx].count > tail.count)
                {
                    _sortedNodes[idx + 1] = _sortedNodes[idx];
                    idx--;
                }
                _sortedNodes[idx + 1] = tail;
            }
            else // newly created node has the smallest count
            {
                _sortedNodes[--_sortedNodesHead] = nodeTree[_treeIndex];
                _sortedNodes[_sortedNodesHead].left = _treeIndex;
            }
        }

        for (uint16_t i = ++_sortedNodesHead; i <= _sortedNodesTail; i++)
        {
            nodeTree[_treeIndex + 1].count = _sortedNodes[i].count + nodeTree[_treeIndex].count;
            nodeTree[_treeIndex + 1].left = _sortedNodes[i].left;
            nodeTree[_treeIndex + 1].right = _treeIndex;
            _treeIndex++;
        }
    }

    DEBUG_PRINT("Huffman tree built");
}

void Compressor::populateCodeTable()
{
    DEBUG_PRINT("Populating code table");

    #if __AVX512F__
        __m512i codes = _mm512_set1_epi32(1); // set all codes to 1 (1 stands for an empty code)
        #pragma GCC unroll NUMBER_OF_SYMBOLS / 16
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i += 16)
        {
            _mm512_storeu_si512(_codeTable + i, codes);
        }

        #pragma GCC unroll MAX_CODE_LENGTH / 2
        for (uint8_t i = 0; i < MAX_CODE_LENGTH / 2; i++)
        {
            reinterpret_cast<uint64v8_t *>(_symbolsAtDepths)[i] = _mm512_setzero_si512();
        }
    #else
        __m256i codes = _mm256_set1_epi32(1); // set all codes to 1 (1 stands for an empty code)
        #pragma GCC unroll NUMBER_OF_SYMBOLS / 8
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i += 8)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(_codeTable + i), codes);
        }

        #pragma GCC unroll MAX_CODE_LENGTH
        for (uint8_t i = 0; i < MAX_CODE_LENGTH; i++)
        {
            _symbolsAtDepths[i] = _mm256_setzero_si256();
        }
    #endif

    uint32_t lastCode = 0;
    uint32_t delta = 0;
    queue<Node> nodeQueue;       // TODO optimize with array
    _tree[_treeIndex].count = 0; // use the count as depth counter
    nodeQueue.push(_tree[_treeIndex]);

    uint32_t lastDepth = 0;
    _usedDepths = 0;
    while (!nodeQueue.empty())
    {
        Node node = nodeQueue.front();
        if (node.isLeaf())
        {
            delta = node.count - lastDepth;
            lastDepth = node.count;
            lastCode = (lastCode + 1) << delta;
            if (delta)
            {
                cerr << "Delta: " << delta << " Last code: " << bitset<16>(lastCode) << " length: " << lastDepth << endl;
            }
            _codeTable[node.right] = lastCode;

            uint64_t bits[4];
            reinterpret_cast<uint64v4_t *>(bits)[0] = _mm256_setzero_si256();
            //vectorBits = _mm256_insert_epi64(vectorBits, 1UL << node.right, 0);
            _usedDepths |= 1UL << node.count;
            node.right = 255 - node.right;
            bits[3] = (uint64_t)(node.right < 64) << node.right;
            bits[2] = (uint64_t)(node.right < 128 && node.right >= 64) << (node.right - 64);
            bits[1] = (uint64_t)(node.right < 192 && node.right >= 128) << (node.right - 128);
            bits[0] = (uint64_t)(node.right >= 192) << (node.right - 192);
            _symbolsAtDepths[node.count] = _mm256_or_si256(_symbolsAtDepths[node.count], reinterpret_cast<uint64v4_t *>(bits)[0]);
        }
        else
        {
            _tree[node.left].count = node.count + 1;
            _tree[node.right].count = node.count + 1;
            nodeQueue.push(_tree[node.left]);
            nodeQueue.push(_tree[node.right]);
        }
        nodeQueue.pop();
    }

    _longestCode = 31 - countl_zero(lastCode);
    lastCode = 0;
    delta = 0;
    uint8_t depthIndex = 0;
    for (uint8_t i = 0; i < MAX_CODE_LENGTH; i++)
    {
        if (_usedDepths & (1UL << i))
        {
            _symbolsAtDepths[depthIndex++] = _symbolsAtDepths[i];
            uint64v4_t bitsVector = _mm256_load_si256(_symbolsAtDepths + i);
            uint64_t *bits = reinterpret_cast<uint64_t *>(&bitsVector);
            lastCode = (lastCode + 1) << delta;
            lastCode--;
            delta = 0;
            for (uint8_t j = 0; j < 4; j++)
            {
                uint8_t symbol = j << 6;
                uint8_t leadingZeros;
                //cerr << "LZ: " << countl_zero(bits[j]) << endl;
                while ((leadingZeros = countl_zero(bits[j])) < 64)
                {
                    uint8_t adjustedSymbol = symbol + leadingZeros;
                    lastCode++;
                    bits[j] ^= 1UL << (63 - leadingZeros);
                    //cerr << (int)adjustedSymbol << " " << lastCode << " " << bitset<16>(lastCode) << endl;
                    _codeTable[adjustedSymbol] = lastCode;
                }
            }
            bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
            cerr << "Depth: " << (int)i << " Vect: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
        }
        delta++;
    }
    cerr << endl;

    /*for (uint8_t i = 0; i < popcount(_usedDepths); i++)
    {
        uint64v4_t bitsVector = _mm256_load_si256(_symbolsAtDepths + i);
        uint64_t *bits = reinterpret_cast<uint64_t *>(&bitsVector);
        cerr << "Depth: " << (int)i << " Vect: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
    }*/

    for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
    {
        uint16_t code = _codeTable[i];
        uint16_t leadingZeros = countl_zero(code);
        uint16_t codeLength = 15 - leadingZeros;
        bitset<10> codeBits(code);
        cerr << (int)i << ": " << codeBits << " " << codeLength << endl;
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
    compressedData += startingIdx >> 1;

    compressedData[0] = 0;
    uint32_t currentCompressedIdx = 0;
    uint32_t nextCompressedIdx = 1;
    uint32_t sourceDataIdx = 1;
    uint32_t sameSymbolCount = 1;
    symbol_t current = firstSymbol;
    uint8_t chunkIdx = 0;

    while (sourceDataIdx <= bytesToCompress)
    {
        if (sourceData[sourceDataIdx] == current)
        {
            sameSymbolCount++;
        }
        else
        {
            uint16_t code = _codeTable[current];
            uint16_t leadingZeros = countl_zero(code);
            uint16_t codeLength = 15 - leadingZeros;
            uint16_t maskedCode = code << (16 - codeLength);
            //bitset<16> maskedCodeBits(maskedCode);
            //DEBUG_PRINT((char)current << ": " << maskedCodeBits << ", times:" << sameSymbolCount);

            for (uint32_t i = 0; i < sameSymbolCount && i < 3; i++)
            {
                uint16_t downShiftedCode = maskedCode >> chunkIdx;
                uint16_t upShiftedCode = maskedCode << (16 - chunkIdx);
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
                uint16_t count = sameSymbolCount & 0x7F;
                sameSymbolCount >>= (BITS_PER_REPETITION_NUMBER - 1);
                repeating = sameSymbolCount > 0;
                count |= repeating << (BITS_PER_REPETITION_NUMBER - 1);
                count <<= BITS_PER_REPETITION_NUMBER;
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
    
    if (_size < _headerSize + _threadBlocksSizesSize + sizeof(uint64v4_t) * popcount(_usedDepths) + _compressedSizesExScan[_numberOfThreads]) // data not compressed
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

    #ifdef _DEBUG_PRINT_ACTIVE_
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

        cerr << "Block types: ";
        for (uint16_t i = 0; i < _blockTypesByteSize; i++)
        {
            cerr << bitset<8>(_blockTypes[i]) << " ";
        }
        cerr << endl;

        cerr << "Compressed data: ";
        for (uint16_t i = 0; i < 20; i++)
        {
            cerr << bitset<8>(_destinationBuffer[i]) << " ";
        }
        cerr << endl;
    #endif
    }
    outputFile.close();

    DEBUG_PRINT("Output file written");
}

void Compressor::printTree(uint16_t nodeIdx, uint16_t indent = 0)
{
#ifdef _DEBUG_PRINT_ACTIVE_
    if (_tree[nodeIdx].isLeaf())
    {
        Leaf leaf = reinterpret_cast<Leaf *>(_tree)[nodeIdx];
        cerr << string(indent, ' ') <<  nodeIdx << ": Leaf: " << leaf.count << " " << (int)(leaf.symbol) << endl;
    }
    else
    {
        cerr << string(indent, ' ') <<  nodeIdx << ": Node: " << _tree[nodeIdx].count << " " << _tree[nodeIdx].left << " " << _tree[nodeIdx].right << endl;
        printTree(_tree[nodeIdx].left, indent + 2);
        printTree(_tree[nodeIdx].right, indent + 2);
    }
#endif
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
    #pragma omp single
    {
        DEBUG_PRINT("Static compression started");

        computeHistogram(); // TODO: parallelize

        // sort the histogram by frequency of symbols in ascending order
        sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, 
            [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
        
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
            sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, 
                 [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
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
