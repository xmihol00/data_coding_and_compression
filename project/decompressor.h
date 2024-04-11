#ifndef _DECOMPRESSOR_H_
#define _DECOMPRESSOR_H_

#include <string>

class Decompressor
{
public:
    Decompressor() = default;
    ~Decompressor() = default;
    void decompress(std::string inputFileName, std::string outputFileName);
};

#endif