#include "compressor.h"

using namespace std;

Compressor::Compressor(bool model, bool adaptive, uint32_t width)
    : _model(model), _adaptive(adaptive), _width(width) { }

Compressor::~Compressor()
{ 
    if (_image != nullptr)
    {
        free(_image);
        _image = nullptr;
    }
}

void Compressor::compress(string inputFileName, string outputFileName)
{
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

    _height = _size / _width;
    if (_height * _width != _size) // verify that file is rectangular
    {
        cerr << "Error: The input file size is not a multiple of the specified width." << endl;
        exit(1);
    }

    _image = static_cast<pixel_t *>(aligned_alloc(64, _width * _height * sizeof(pixel_t)));
    if (_image == nullptr)
    {
        cerr << "Error: Unable to allocate memory for the image." << endl;
        exit(1);
    }

    inputFile.read(reinterpret_cast<char *>(_image), _size);
}
