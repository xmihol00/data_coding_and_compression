#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint64_t width) 
    : HuffmanRLECompression(model, adaptive, width) { }

Compressor::~Compressor()
{ 
    if (_dataPool != nullptr)
    {
        free(_dataPool);
    }

    if (_bestBlockTraversals != nullptr)
    {
        delete[] _bestBlockTraversals;
    }

    for (uint8_t i = 0; i < MAX_NUM_THREADS; i++)
    {
        if (_rlePerBlockCounts[i] != nullptr)
        {
            delete[] _rlePerBlockCounts[i];
        }
    }

    if (_serializedData != nullptr)
    {
        free(_serializedData);
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
        _histogram[_fileData[i]].count++;
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

    // set all codes to 1 (code boundary)
    #if __AVX512F__
        __m512i codes = _mm512_set1_epi32(1);
        #pragma GCC unroll NUMBER_OF_SYMBOLS / 16
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i += 16)
        {
            _mm512_storeu_si512(_codeTable + i, codes);
        }
        for (uint8_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
        {
            reinterpret_cast<uint64v8_t *>(_symbolsAtDepths)[i] = _mm512_setzero_si512();
        }
    #else
        __m256i codes = _mm256_set1_epi32(1);
        #pragma GCC unroll NUMBER_OF_SYMBOLS / 8
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i += 8)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(_codeTable + i), codes);
        }
        for (uint8_t i = 0; i < MAX_LONG_CODE_LENGTH; i++)
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
    for (uint8_t i = 0; i < MAX_LONG_CODE_LENGTH; i++)
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
            cerr << "Depth: " << (int)i << " Bits: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
            bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
            cerr << "Depth: " << (int)i << " Vect: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
        }
        delta++;
    }
    cerr << endl;

    for (uint8_t i = 0; i < popcount(_usedDepths); i++)
    {
        uint64v4_t bitsVector = _mm256_load_si256(_symbolsAtDepths + i);
        uint64_t *bits = reinterpret_cast<uint64_t *>(&bitsVector);
        cerr << "Depth: " << (int)i << " Bits: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
    }

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

void Compressor::transformRLE()
{
    DEBUG_PRINT("Transforming RLE");

    _compressedData[0] = 0;
    uint32_t currentCompressedIndex = 0;
    uint32_t nextCompressedIndex = 1;
    uint32_t imageIndex = 1;
    uint32_t sameSymbolCount = 1;
    symbol_t current = _fileData[0];
    uint8_t chunkIndex = 0;

    if (_longestCode >= 16)
    {
        while (imageIndex <= _size)
        {
            if (_fileData[imageIndex] == current)
            {
                sameSymbolCount++;
            }
            else
            {
                uint32_t code = _codeTable[current];
                uint32_t leadingZeros = countl_zero(code);
                uint32_t codeLength = 31 - leadingZeros;
                uint32_t mask = ~(1 << codeLength);
                uint32_t maskedCode = code & mask;
                bitset<32> maskedCodeBits(maskedCode);
                cerr << (int)current << ": " << maskedCodeBits << ", times:" << sameSymbolCount << endl;

                for (uint32_t i = 0; i < sameSymbolCount && i < 3; i++)
                {
                    chunkIndex += codeLength;
                    uint32_t upShiftedCode = maskedCode << (32 - chunkIndex);
                    uint32_t downShiftedCode = maskedCode >> chunkIndex;
                    _compressedData[currentCompressedIndex] |= upShiftedCode;
                    _compressedData[nextCompressedIndex] = 0;
                    _compressedData[nextCompressedIndex] |= downShiftedCode;
                    bool moveChunk = chunkIndex >= 32;
                    chunkIndex &= 31;
                    currentCompressedIndex += moveChunk;
                    nextCompressedIndex += moveChunk;
                }

                bool repeating = sameSymbolCount >= 3;
                sameSymbolCount -= 3;
                while (repeating)
                {
                    uint32_t count = sameSymbolCount & 0x7F;
                    sameSymbolCount >>= 7;
                    repeating = sameSymbolCount > 0;
                    count |= repeating << 7;
                    chunkIndex += 8;
                    uint32_t upShiftedCount = count << chunkIndex;
                    uint32_t downShiftedCount = count >> (32 - chunkIndex);
                    _compressedData[currentCompressedIndex] |= upShiftedCount;
                    _compressedData[nextCompressedIndex] = 0;
                    _compressedData[nextCompressedIndex] |= downShiftedCount;
                    bool moveChunk = chunkIndex >= 32;
                    chunkIndex &= 31;
                    currentCompressedIndex += moveChunk;
                    nextCompressedIndex += moveChunk;
                }

                sameSymbolCount = 1;
                current = _fileData[imageIndex];
            }
            imageIndex++;
        }
    }
    else
    {
        uint16_t *compressedData = reinterpret_cast<uint16_t *>(_compressedData);
        while (imageIndex <= _size)
        {
            if (_fileData[imageIndex] == current)
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
                //cerr << (int)current << ": " << maskedCodeBits << ", times:" << sameSymbolCount << endl;

                for (uint32_t i = 0; i < sameSymbolCount && i < 3; i++)
                {
                    uint16_t downShiftedCode = maskedCode >> chunkIndex;
                    uint16_t upShiftedCode = maskedCode << (16 - chunkIndex);
                    compressedData[currentCompressedIndex] |= downShiftedCode;
                    compressedData[nextCompressedIndex] = 0;
                    compressedData[nextCompressedIndex] |= upShiftedCode;
                    chunkIndex += codeLength;
                    bool moveChunk = chunkIndex >= 16;
                    chunkIndex &= 15;
                    currentCompressedIndex += moveChunk;
                    nextCompressedIndex += moveChunk;
                }

                bool repeating = sameSymbolCount >= 3;
                sameSymbolCount -= 3;
                while (repeating)
                {
                    uint16_t count = sameSymbolCount & 0x7F;
                    sameSymbolCount >>= 7;
                    repeating = sameSymbolCount > 0;
                    count |= repeating << 7;
                    count <<= 8;
                    uint16_t downShiftedCount = count >> chunkIndex;
                    uint16_t upShiftedCount = count << (16 - chunkIndex);
                    compressedData[currentCompressedIndex] |= downShiftedCount;
                    compressedData[nextCompressedIndex] = 0;
                    compressedData[nextCompressedIndex] |= upShiftedCount;
                    chunkIndex += 8;
                    bool moveChunk = chunkIndex >= 16;
                    chunkIndex &= 15;
                    currentCompressedIndex += moveChunk;
                    nextCompressedIndex += moveChunk;
                }

                sameSymbolCount = 1;
                current = _fileData[imageIndex];
            }
            imageIndex++;
        }
    }
    _compressedSize = nextCompressedIndex;

    DEBUG_PRINT("RLE transformed");
}

void Compressor::createHeader()
{
    DEBUG_PRINT("Creating header");

    if (_longestCode >= 16)
    {
        _headerSize = sizeof(CodeLengthsHeader);
        CodeLengthsHeader &header = reinterpret_cast<CodeLengthsHeader &>(_header);
        header.width = static_cast<uint32_t>(_width);
        header.blockSize = static_cast<uint32_t>(_size);
        header.headerType = HEADERS.STATIC | HEADERS.DIRECT | HEADERS.ALL_SYMBOLS | HEADERS.CODE_LENGTHS_32;
        header.version = 0;
        for (uint16_t i = 0; i < NUMBER_OF_SYMBOLS; i++)
        {
            // TODO: fill in the header
        }
    }
    else
    {
        /*CodeLengthsHeader &header = reinterpret_cast<CodeLengthsHeader &>(_header);
        header.width = static_cast<uint32_t>(_width);
        header.blockSize = static_cast<uint32_t>(_size);
        header.headerType = HEADERS.STATIC | HEADERS.DIRECT | HEADERS.ALL_SYMBOLS | HEADERS.CODE_LENGTHS_16;
        header.version = 0;

        _headerSize = sizeof(BaseHeader) + ADDITIONAL_HEADER_SIZES[header.headerType];

        for (uint16_t i = 0, j = 0; i < NUMBER_OF_SYMBOLS; i += 2, j++)
        {
            header.codeLengths[j] = (31 - countl_zero(_codeTable[i])) << 4;
            header.codeLengths[j] |= 31 - countl_zero(_codeTable[i + 1]);
        }*/

        DepthBitmapsHeader &header = reinterpret_cast<DepthBitmapsHeader &>(_header);
        header.width = static_cast<uint32_t>(_width);
        header.blockSize = static_cast<uint32_t>(_size);
        header.headerType = HEADERS.STATIC | HEADERS.DIRECT | HEADERS.ALL_SYMBOLS | HEADERS.CODE_LENGTHS_16;
        header.version = 0;
        header.codeDepths = _usedDepths;
        _headerSize = sizeof(DepthBitmapsHeader);
    }

    DEBUG_PRINT("Header created");
}

void Compressor::writeOutputFile(std::string outputFileName)
{
    DEBUG_PRINT("Writing output file");

    ofstream outputFile(outputFileName, ios::binary);
    if (!outputFile.is_open())
    {
        cerr << "Error: Unable to open output file '" << outputFileName << "'." << endl;
        exit(1);
    }
    
    outputFile.write(reinterpret_cast<char *>(_serializedData), _size); // TODO
    /*outputFile.write(reinterpret_cast<char *>(&_header), _headerSize);
    outputFile.write(reinterpret_cast<char *>(_symbolsAtDepths), sizeof(uint64v4_t) * popcount(_usedDepths));
    outputFile.write(reinterpret_cast<char *>(_compressedData), _compressedSize * sizeof(uint32_t)); // TODO remove last up to 3 bytes */
    outputFile.close();

    DEBUG_PRINT("Output file written");
}

void Compressor::printTree(uint16_t nodeIdx, uint16_t indent = 0)
{
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
}

void Compressor::readInputFile(std::string inputFileName)
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

    _dataPool = static_cast<symbol_t *>(aligned_alloc(64, _width * _height * sizeof(symbol_t) + COMPRESSED_ORIGINAL_OFFSET + 1));
    if (_dataPool == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the image." << endl;
        exit(1);
    }
    // alias the allocated memory with offset to save memory when compressing (already consumed file data can be overwritten by the compressed data)
    _compressedData = reinterpret_cast<uint32_t *>(_dataPool);
    _fileData = _dataPool + COMPRESSED_ORIGINAL_OFFSET;

    inputFile.read(reinterpret_cast<char *>(_fileData), _size);
    inputFile.close();
    _fileData[_size] = ~_fileData[_size - 1]; // add a dummy symbol to ensure correct RLE encoding
}

void Compressor::compressStatic()
{
    computeHistogram();

    // sort the histogram by frequency of symbols in ascending order
    sort(_histogram, _histogram + NUMBER_OF_SYMBOLS, 
         [](const Leaf &a, const Leaf &b) { return a.count < b.count; });
    
    buildHuffmanTree();

    populateCodeTable();
    
    printTree(_treeIndex);

    transformRLE();

    createHeader();

    /*uint16_t *compressedData = reinterpret_cast<uint16_t *>(_compressedData);
    for (uint32_t i = 0; i < _compressedSize && i < 100; i++)
    {
        bitset<16> codeBitset(compressedData[i]);
        cerr << codeBitset << " ";
    }
    cerr << endl;*/
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
                            bool sameSymbol = _fileData[firstColumn + l] == lastSymbol;
                            lastSymbol = _fileData[firstColumn + l];
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
                            bool sameSymbol = _fileData[column] == lastSymbol;
                            lastSymbol = _fileData[column];
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

void Compressor::compressAdaptive()
{
    #pragma omp master
    {
        DEBUG_PRINT("Adaptive compression started");
    }

    analyzeImageAdaptive();
    #pragma omp barrier
    
    #pragma omp single
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
                switch (_bestBlockTraversals[i])
                {
                case HORIZONTAL:
                    // nothing to do here, already serialized
                    break;
                
                case VERTICAL:
                    #pragma omp task firstprivate(j, i)
                    {
                        cerr << "Transposing block " << j << " " << i << " at " << (void *)_fileData << " in thread " << omp_get_thread_num() << endl;
                        transposeSerializeBlock(_fileData, _serializedData, j, i);
                    }
                    break;
                }
            }
        }
    }
    #pragma omp taskwait

    cerr << "Adaptive compression not implemented. " << omp_get_thread_num() << endl;
}

void Compressor::compress(string inputFileName, string outputFileName)
{
    readInputFile(inputFileName);
    
    if (_model)
    {
        if (_adaptive)
        {
            //compressAdaptiveModel();
        }
        else
        {
            //compressStaticModel();
        }
    }
    else
    {
        if (_adaptive)
        {
            _blocksPerRow = (_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
            _blocksPerColumn = (_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
            _blockCount = _blocksPerRow * _blocksPerColumn;
            _bestBlockTraversals = new AdaptiveTraversals[_blockCount];
            _serializedData = reinterpret_cast<symbol_t *>(aligned_alloc(64, _width * _height * sizeof(symbol_t) + 1));
            if (_serializedData == nullptr)
            {
                cerr << "Error: Unable to allocate memory for the image data." << endl;
                exit(1);
            }

            #pragma omp parallel
            compressAdaptive();
        }
        else
        {
            compressStatic();
        }
    }

    writeOutputFile(outputFileName);
}
