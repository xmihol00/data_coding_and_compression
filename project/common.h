#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

constexpr struct 
{
    uint8_t SERIALIZATION    = 0;
    uint8_t STATIC           = 0 << this->SERIALIZATION;
    uint8_t ADAPTIVE         = 1 << this->SERIALIZATION;

    uint8_t TRANSFORMATION   = 1;
    uint8_t DIRECT           = 0 << this->TRANSFORMATION;
    uint8_t MODEL            = 1 << this->TRANSFORMATION;

    uint8_t CODE_TABLE_TYPE  = 2;
    uint8_t ALL_SYMBOLS      = 0 << this->CODE_TABLE_TYPE;
    uint8_t SELECTED_SYMBOLS = 1 << this->CODE_TABLE_TYPE;

    uint8_t CODE_LENGTHS     = 3;
    uint8_t CODE_LENGTHS_16  = 0 << this->CODE_LENGTHS;
    uint8_t CODE_LENGTHS_32  = 1 << this->CODE_LENGTHS;
} HEADER_OPTIONS;

struct BaseHeader
{
    uint32_t width;
    uint32_t blockSize;
    uint8_t headerType;
    uint8_t version;
};

struct StaticHeader16 : public BaseHeader
{
    uint8_t codeLengths[128];
};

struct StaticHeader32 : public BaseHeader
{
    uint8_t codeLengths[160];
};


#endif // _COMMON_H_