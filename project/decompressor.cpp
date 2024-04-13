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
            CodeLengthsHeader &header = reinterpret_cast<CodeLengthsHeader &>(_header); 
            for (uint16_t i = 0, j = 0; i < NUMBER_OF_SYMBOLS / 2; i++, j += 2)
            {
                _codeLengthsSymbols[j].codeLength = _header.buffer[i] >> 4;
                _codeLengthsSymbols[j].symbol = j;
                _codeLengthsSymbols[j + 1].codeLength = _header.buffer[i] & 0x0F;
                _codeLengthsSymbols[j + 1].symbol = j + 1;
            }

            sort(_codeLengthsSymbols, _codeLengthsSymbols + NUMBER_OF_SYMBOLS, 
                 [](const CodeLengthSymbol &a, const CodeLengthSymbol &b) { return a.codeLength < b.codeLength; });
            
            uint16_t code = -1;
            uint8_t lastCodeLength = 0;
            break;
        
        default:
            cout << "Error: Unsupported header type" << endl;
            exit(1);
    }
}

void Decompressor::decompressStatic()
{
    // decompress static
}

void Decompressor::decompress(string inputFileName, string outputFileName)
{
    readInputFile(inputFileName);
    decompressStatic();
    //writeOutputFile(outputFileName);
}