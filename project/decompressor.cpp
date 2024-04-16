#include "decompressor.h"

using namespace std;

Decompressor::~Decompressor()
{
    if (_decompressedData != nullptr)
    {
        free(_decompressedData);
    }
}

void Decompressor::readInputFile(string inputFileName)
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

    /*inputFile.read(reinterpret_cast<char *>(&_header), sizeof(BaseHeader));
    // TODO check headers

    fileSize -= sizeof(BaseHeader) + ADDITIONAL_HEADER_SIZES[_header.headerType];
    inputFile.read(reinterpret_cast<char *>(_header.buffer), ADDITIONAL_HEADER_SIZES[_header.headerType]);*/

    inputFile.read(reinterpret_cast<char *>(&_header), sizeof(DepthBitmapsHeader));
    _usedDepths = reinterpret_cast<DepthBitmapsHeader &>(_header).codeDepths;
    uint16_t additionalHeaderSize = sizeof(uint64v4_t) * popcount(_usedDepths);
    fileSize -= sizeof(DepthBitmapsHeader) + additionalHeaderSize;
    inputFile.read(reinterpret_cast<char *>(_symbolsAtDepths), additionalHeaderSize);

    for (uint8_t i = 0; i < popcount(_usedDepths); i++)
    {
        uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + i);
        cerr << "Bits: " << bitset<64>(bits[0]) << " " << bitset<64>(bits[1]) << " " << bitset<64>(bits[2]) << " " << bitset<64>(bits[3]) << endl;
    }

    _compressedData = reinterpret_cast<uint32_t *>(aligned_alloc(64, fileSize + 64));
    _decompressedData = reinterpret_cast<uint8_t *>(aligned_alloc(64, _header.blockSize + 64));
    if (_compressedData == nullptr || _decompressedData == nullptr)
    {
        cout << "Error: Could not allocate memory for data decompression" << endl;
        exit(1);
    }

    inputFile.read(reinterpret_cast<char *>(_compressedData), fileSize);
}

void Decompressor::parseHeader()
{
    switch (_header.headerType)
    {
        case HEADERS.STATIC | HEADERS.DIRECT | HEADERS.ALL_SYMBOLS | HEADERS.CODE_LENGTHS_16:
            {
            #if __AVX512BW__ && __AVX512VL__
                reinterpret_cast<uint32v16_t *>(_indexPrefixLengthCodeCount)[0] = _mm512_setzero_si512();
                reinterpret_cast<uint32v16_t *>(_indexPrefixLengthCodeCount)[1] = _mm512_setzero_si512();
            #endif

                uint8_t depthIdx = 0;
                //uint8_t depthsIndicesIdx = 0;
                for (uint8_t i = 0; i < MAX_LONG_CODE_LENGTH; i++)
                {
                    if (_usedDepths & (1UL << i))
                    {
                        uint64v4_t popCounts = _mm256_popcnt_epi64(_symbolsAtDepths[depthIdx]); // AVX512VPOPCNTDQ
                        uint64_t *bits = reinterpret_cast<uint64_t *>(&popCounts);
                        uint16_t sum1 = bits[0] + bits[1];
                        uint16_t sum2 = bits[2] + bits[3];
                        uint16_t symbolCount = sum1 + sum2;

                        _depthsIndices[depthIdx].depth = i;
                        _depthsIndices[depthIdx].prefixLength = i - 16 + countl_zero(static_cast<uint16_t>(symbolCount - 1));
                        _depthsIndices[depthIdx].symbolsAtDepthIndex = depthIdx;

                        // get the correct masks index (longest prefix length must have the smallest index)
                        int8_t j = depthIdx - 1;
                        uint8_t currentMasksIdx = depthIdx;
                        for (; j >= 0 && _depthsIndices[j].prefixLength < _depthsIndices[depthIdx].prefixLength; j--)
                        {
                            uint8_t masksIdx = _depthsIndices[j].masksIndex;
                            _depthsIndices[j].masksIndex = currentMasksIdx;
                            currentMasksIdx = masksIdx;
                        }
                        _depthsIndices[depthIdx].masksIndex = currentMasksIdx;

                        depthIdx++;    
                    }
                }

                uint8_t lastDepth = 0;
                uint16_t lastCode = -1;
                uint16_t symbolIdx = 0;
                for (uint8_t i = 0; i < depthIdx; i++)
                {
                    cerr << "Depth: " << (int)_depthsIndices[i].depth << " PrefixLength: " << (int)_depthsIndices[i].prefixLength << " SymbolsAtDepthIndex: " << (int)_depthsIndices[i].symbolsAtDepthIndex << " MasksIndex: " << (int)_depthsIndices[i].masksIndex << endl;

                    uint8_t delta = _depthsIndices[i].depth - lastDepth;
                    lastDepth = _depthsIndices[i].depth;
                    cerr << "Delta: " << (int)delta << endl;
                    lastCode = (lastCode + 1) << delta;
                    cerr << "LastCode: " << (int)lastCode << endl;
                    _codePrefixesSmall[15 - _depthsIndices[i].masksIndex] = lastCode << (16 - _depthsIndices[i].depth);
                    _codeMasksSmall[15 - _depthsIndices[i].masksIndex] = (~0U) << (16 - _depthsIndices[i].prefixLength);
                    _prefixIndices[_depthsIndices[i].masksIndex] = symbolIdx;
                    _prefixShifts[_depthsIndices[i].masksIndex] = _depthsIndices[i].prefixLength;
                    _suffixShifts[_depthsIndices[i].masksIndex] = 16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength;
                    symbolIdx += lastCode & (static_cast<uint16_t>(~0) >> (16 - _depthsIndices[i].depth + _depthsIndices[i].prefixLength));

                    uint16_t lastSymbolIdx = symbolIdx;
                    uint64_t *bits = reinterpret_cast<uint64_t *>(_symbolsAtDepths + _depthsIndices[i].symbolsAtDepthIndex);
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
                }

                /*CodeLengthsHeader &header = reinterpret_cast<CodeLengthsHeader &>(_header);
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
                for (uint8_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
                {
                    _indexPrefixLengthCodeCount[i].cumulativeCodeCount = lastCodeCount;
                    lastCodeCount += _indexPrefixLengthCodeCount[i].codeCount; 
                    _indexPrefixLengths[i].index = i;
                    _indexPrefixLengths[i].prefixLength = (i - 16 + countl_zero(static_cast<uint16_t>(_indexPrefixLengthCodeCount[i].codeCount - 1)));
                }
                sort(_indexPrefixLengths, _indexPrefixLengths + MAX_SHORT_CODE_LENGTH, 
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
                }*/
                _codePrefixesSmallVector[0] = _mm256_and_si256(_codeMasksSmallVector[0], _codePrefixesSmallVector[0]);
                
                for (uint16_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
                {
                    bitset<16> prefix(_codePrefixesSmall[i]);
                    bitset<16> mask(_codeMasksSmall[i]);
                    cerr << i << ": " << prefix << " " << mask << " " << (int)_prefixShifts[i] << " " << (int)_suffixShifts[i] << " " << (int)_prefixIndices[i] << endl;
                }
                exit(0);
            }
            break;
        
        default:
            cout << "Error: Unsupported header type" << endl;
            exit(1);
    }
}

void Decompressor::decompressStatic()
{
    parseHeader();
#if __AVX512BW__ && __AVX512VL__
    uint16_t *compressedData = reinterpret_cast<uint16_t *>(_compressedData);
    bool overflow;
    uint32_t decompressedIdx = 0;
    uint32_t currentIdx = 0;
    uint32_t nextIdx = 1;
    uint16_t bitLength = 0;
    uint16_t inverseBitLength = 16;
    uint16v16_t prefixes = _mm256_load_si256(_codePrefixesSmallVector);
    uint16v16_t masks = _mm256_load_si256(_codeMasksSmallVector);
    uint16_t current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
    uint16v16_t currentVector = _mm256_set1_epi16(current);
    uint16v16_t masked = _mm256_and_si256(currentVector, masks);
    uint16v16_t xored = _mm256_xor_si256(prefixes, masked);
    uint16_t prefixBitMap = _mm256_cmp_epu16_mask(xored, _mm256_setzero_si256(), _MM_CMPINT_EQ);
    cerr << "PrefixBitMap: " << bitset<16>(prefixBitMap) << endl;
    uint8_t prefixIdx = countl_zero(prefixBitMap);
    cerr << "PrefixIdx: " << (int)prefixIdx << endl;
    uint8_t prefixLength = _prefixShifts[prefixIdx];
    cerr << "PrefixLength: " << (int)prefixLength << endl;
    uint8_t suffixShift = _suffixShifts[prefixIdx];
    cerr << "SuffixShift: " << (int)suffixShift << endl;
    uint8_t suffix = static_cast<uint16_t>(current << prefixLength) >> suffixShift;
    uint8_t codeLength = 16 + prefixLength - suffixShift;
    symbol_t symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
    _decompressedData[decompressedIdx++] = symbol;
    cerr << "Symbol: " << (char)symbol << " " << (int)symbol << "\n" << endl;
    bitLength += codeLength;
    symbol_t lastSymbol = symbol;
    uint8_t sameSymbolCount = 0;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < _header.blockSize)
    {
        //cerr << "BitLength: " << (int)bitLength << endl;
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
        //cerr << "Current: " << bitset<16>(current) << endl;
        currentVector = _mm256_set1_epi16(current);
        masked = _mm256_and_si256(currentVector, masks);
        xored = _mm256_xor_si256(prefixes, masked);
        prefixBitMap = _mm256_cmp_epu16_mask(xored, _mm256_setzero_si256(), _MM_CMPINT_EQ);
        //cerr << "PrefixBitMap: " << bitset<16>(prefixBitMap) << endl;
        prefixIdx = countl_zero(prefixBitMap);
        //cerr << "PrefixIdx: " << (int)prefixIdx << endl;
        prefixLength = _prefixShifts[prefixIdx];
        //cerr << "PrefixLength: " << (int)prefixLength << endl;
        suffixShift = _suffixShifts[prefixIdx];
        //cerr << "SuffixShift: " << (int)suffixShift << endl;
        suffix = static_cast<uint16_t>(current << prefixLength) >> suffixShift;
        codeLength = 16 + prefixLength - suffixShift;
        symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
        //cerr << "Suffix: " << (int)suffix << endl;
        cerr << "Symbol: " << (char)symbol << " " << (int)symbol << "\n" << endl;
        _decompressedData[decompressedIdx++] = symbol;
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
            constexpr uint8_t repetition_bits = 8; // TODO move to header
            bool repeat;
            uint8_t multiplier = 0;
            do
            {
                current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
                cerr << "Current: " << bitset<16>(current) << endl;
                repeat = current & 0x8000;
                uint32_t repetitions = ((current & 0x7FFF) >> repetition_bits);
                repetitions <<= multiplier;
                cerr << "Repetitions: " << repetitions << endl;

                for (uint32_t i = 0; i < repetitions; i++)
                {
                    _decompressedData[decompressedIdx++] = symbol;
                }

                bitLength += repetition_bits;
                overflow = bitLength >= 16;
                bitLength &= 0x0F;
                nextIdx += overflow;
                currentIdx += overflow;
                inverseBitLength = 16 - bitLength;
                multiplier += (repetition_bits - 1);
            }
            while (repeat);
            cerr << endl;
        }

        /*if (nextIdx >= 5)
        {
            exit(0);
        }*/
    }
    cerr << "DecompressedIdx: " << decompressedIdx << " block size:" << _header.blockSize << endl;
#endif
}

void Decompressor::writeOutputFile(std::string outputFileName)
{
    ofstream outputFile(outputFileName, ios::binary);
    if (!outputFile.is_open())
    {
        cout << "Error: Could not open output file " << outputFileName << endl;
        exit(1);
    }

    outputFile.write(reinterpret_cast<char *>(_decompressedData), _header.blockSize);
    outputFile.close();
}

void Decompressor::decompress(string inputFileName, string outputFileName)
{
    readInputFile(inputFileName);
    decompressStatic();
    writeOutputFile(outputFileName);
}
