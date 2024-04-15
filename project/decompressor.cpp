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

    inputFile.read(reinterpret_cast<char *>(&_header), sizeof(BaseHeader));
    // TODO check headers

    fileSize -= sizeof(BaseHeader) + ADDITIONAL_HEADER_SIZES[_header.headerType];
    inputFile.read(reinterpret_cast<char *>(_header.buffer), ADDITIONAL_HEADER_SIZES[_header.headerType]);

    _compressedData = reinterpret_cast<uint32_t *>(aligned_alloc(64, fileSize));
    _decompressedData = reinterpret_cast<uint8_t *>(aligned_alloc(64, _header.blockSize));
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
                CodeLengthsHeader &header = reinterpret_cast<CodeLengthsHeader &>(_header);
                for (uint16_t i = 0, j = 0; i < NUMBER_OF_SYMBOLS / 2; i++, j += 2)
                {
                    _codeLengthsSymbols[j].codeLength = header.codeLengths[i] >> 4;
                    _codeLengthsSymbols[j].symbol = j;
                    _codeLengthsSymbols[j + 1].codeLength = header.codeLengths[i] & 0x0F;
                    _codeLengthsSymbols[j + 1].symbol = j + 1;
                }

                sort(_codeLengthsSymbols, _codeLengthsSymbols + NUMBER_OF_SYMBOLS, 
                     [](const CodeLengthSymbol &a, const CodeLengthSymbol &b) 
                     { return a.codeLength < b.codeLength || (a.codeLength == b.codeLength && a.symbol < b.symbol); });

                uint16_t codesIdx = 0;
                while (_codeLengthsSymbols[codesIdx].codeLength == 0)
                {
                    codesIdx++;
                }

                uint16_t lastCode = -1;
                uint16_t samePrefixCount = 0;
                uint8_t lastCodeLength = 0;
                uint8_t maskIdx = 0;
                uint8_t prefixIdx = 16;
                constexpr uint8_t I = 1;

                for (uint16_t i = 0; codesIdx < NUMBER_OF_SYMBOLS; codesIdx++, i++)
                {
                    uint8_t delta = _codeLengthsSymbols[codesIdx].codeLength - lastCodeLength;
                    lastCode = (lastCode + 1) << delta;

                    // overwrite the currently stored values in all lookup tables (some delayed by one iteration)
                    _codePrefixesSmall[maskIdx] = lastCode << (16 - _codeLengthsSymbols[codesIdx].codeLength);
                    _codeMasksSmall[maskIdx - I] = (~0U) << (32 - lastCodeLength - countl_zero(samePrefixCount));
                    _prefixIndices[prefixIdx - I] = i;
                    _prefixLengths[prefixIdx] = popcount(_codeMasksSmall[maskIdx - I]);
                    _suffixShifts[prefixIdx] = 16 - lastCodeLength + _prefixLengths[prefixIdx];

                    // store the symbol at its sorted position by the code length
                    _symbolsTable[i] = _codeLengthsSymbols[codesIdx].symbol;

                    // update state based on the delta, i.e. if the code length has changed
                    bool positiveDelta = delta > 0;
                    lastCodeLength = _codeLengthsSymbols[codesIdx].codeLength;
                    samePrefixCount++;
                    maskIdx += positiveDelta;
                    prefixIdx -= positiveDelta;
                    samePrefixCount *= !positiveDelta; // reset if code length has changed
                }
                // fill in the delayed lookup table entries
                uint16_t mask = (~0U) << (32 - lastCodeLength - countl_zero(samePrefixCount));
                uint8_t prefixLength = popcount(mask);
                uint8_t sufixShift = 16 - lastCodeLength + prefixLength;

                // set the rest of the table to repeat the last entry
                for ( ; maskIdx <= MAX_SHORT_CODE_LENGTH; maskIdx++, prefixIdx--)
                {
                    _codePrefixesSmall[maskIdx] = _codePrefixesSmall[maskIdx - I];
                    _codeMasksSmall[maskIdx - I] = mask;
                    _prefixLengths[prefixIdx] = prefixLength;
                    _suffixShifts[prefixIdx] = sufixShift;
                    _prefixIndices[prefixIdx - I] = _prefixIndices[prefixIdx];
                }
                _codePrefixesSmallVector[0] = _mm256_and_si256(_codeMasksSmallVector[0], _codePrefixesSmallVector[0]);
                
                for (uint16_t i = 0; i < MAX_SHORT_CODE_LENGTH; i++)
                {
                    bitset<16> prefix(_codePrefixesSmall[i]);
                    bitset<16> mask(_codeMasksSmall[i]);
                    cerr << i << ": " << prefix << " " << mask << " " << (int)_prefixLengths[i] << " " << (int)_suffixShifts[i] << " " << (int)_prefixIndices[i] << endl;
                }
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
    uint8_t prefixLength = _prefixLengths[prefixIdx];
    cerr << "PrefixLength: " << (int)prefixLength << endl;
    uint8_t suffixShift = _suffixShifts[prefixIdx];
    cerr << "SuffixShift: " << (int)suffixShift << endl;
    uint8_t suffix = static_cast<uint16_t>(current << prefixLength) >> suffixShift;
    uint8_t codeLength = 16 + prefixLength - suffixShift;
    symbol_t symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
    _decompressedData[decompressedIdx++] = symbol;
    cerr << "Symbol: " << (char)symbol << " " << (int)symbol << endl;
    bitLength += codeLength;
    symbol_t lastSymbol = symbol;
    uint8_t sameSymbolCount = 1;
    inverseBitLength = 16 - bitLength;

    while (decompressedIdx < _header.blockSize)
    {
        //cerr << "BitLength: " << (int)bitLength << endl;
        current = (compressedData[currentIdx] << bitLength) | (compressedData[nextIdx] >> inverseBitLength);
        cerr << "Current: " << bitset<16>(current) << endl;
        currentVector = _mm256_set1_epi16(current);
        masked = _mm256_and_si256(currentVector, masks);
        xored = _mm256_xor_si256(prefixes, masked);
        prefixBitMap = _mm256_cmp_epu16_mask(xored, _mm256_setzero_si256(), _MM_CMPINT_EQ);
        cerr << "PrefixBitMap: " << bitset<16>(prefixBitMap) << endl;
        prefixIdx = countl_zero(prefixBitMap);
        //cerr << "PrefixIdx: " << (int)prefixIdx << endl;
        prefixLength = _prefixLengths[prefixIdx];
        //cerr << "PrefixLength: " << (int)prefixLength << endl;
        suffixShift = _suffixShifts[prefixIdx];
        //cerr << "SuffixShift: " << (int)suffixShift << endl;
        suffix = static_cast<uint16_t>(current << prefixLength) >> suffixShift;
        codeLength = 16 + prefixLength - suffixShift;
        symbol = _symbolsTable[_prefixIndices[prefixIdx] + suffix];
        cerr << "Symbol: " << (char)symbol << " " << (int)symbol << endl;
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
        }

        if (nextIdx >= 5)
        {
            exit(0);
        }
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
