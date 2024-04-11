#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>

using pixel_t = uint8_t;

class Compressor
{
public:
    Compressor(bool model, bool adaptive, uint32_t width);
    ~Compressor();
    void compress(std::string inputFileName, std::string outputFileName);

private:
    void readInputFile(std::string inputFileName);

    bool _model;
    bool _adaptive;
    uint32_t _width;
    uint32_t _height;

    pixel_t *_image = nullptr;
};

#endif