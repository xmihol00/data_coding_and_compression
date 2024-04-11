#ifndef _MAIN_H_
#define _MAIN_H_

#include <iostream>
#include <string>
#include <vector>

struct Arguments
{
    bool compress;
    bool decompress;
    bool model;
    bool adaptive;
    std::string inputFileName;
    std::string outputFileName;
    uint32_t width;
};

Arguments parseArguments(int argc, char* argv[]);

#endif