#ifndef _MAIN_H_
#define _MAIN_H_

#include "omp.h"

#include <iostream>
#include <string>
#include <vector>
#include <bit>

#include "compressor.h"
#include "decompressor.h"

struct Arguments
{
    bool compress;
    bool decompress;
    bool model;
    bool adaptive;
    std::string inputFileName;
    std::string outputFileName;
    uint64_t width;
    uint8_t threads;
};

Arguments parseArguments(int argc, char* argv[]);

#endif