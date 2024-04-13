#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

class HuffmanRLECompression
{
public:
    HuffmanRLECompression() = default;
    ~HuffmanRLECompression() = default;

protected:
    static constexpr uint16_t NUMBER_OF_SYMBOLS{256};

    static constexpr struct
    {
        uint8_t SERIALIZATION   = 0;
        uint8_t TRANSFORMATION  = 1;
        uint8_t CODE_TABLE_TYPE = 2;
        uint8_t CODE_LENGTHS    = 3;

    } HEADER_OPTIONS
    {
        .SERIALIZATION   = 0,
        .TRANSFORMATION  = 1,
        .CODE_TABLE_TYPE = 2,
        .CODE_LENGTHS    = 3,
    };

    static constexpr struct 
    {
        uint8_t STATIC           = 0 << HEADER_OPTIONS.SERIALIZATION;
        uint8_t ADAPTIVE         = 1 << HEADER_OPTIONS.SERIALIZATION;

        uint8_t DIRECT           = 0 << HEADER_OPTIONS.TRANSFORMATION;
        uint8_t MODEL            = 1 << HEADER_OPTIONS.TRANSFORMATION;

        uint8_t ALL_SYMBOLS      = 0 << HEADER_OPTIONS.CODE_TABLE_TYPE;
        uint8_t SELECTED_SYMBOLS = 1 << HEADER_OPTIONS.CODE_TABLE_TYPE;

        uint8_t CODE_LENGTHS_16  = 0 << HEADER_OPTIONS.CODE_LENGTHS;
        uint8_t CODE_LENGTHS_32  = 1 << HEADER_OPTIONS.CODE_LENGTHS;
    } HEADERS
    {
        .STATIC           = 0 << HEADER_OPTIONS.SERIALIZATION,
        .ADAPTIVE         = 1 << HEADER_OPTIONS.SERIALIZATION,

        .DIRECT           = 0 << HEADER_OPTIONS.TRANSFORMATION,
        .MODEL            = 1 << HEADER_OPTIONS.TRANSFORMATION,

        .ALL_SYMBOLS      = 0 << HEADER_OPTIONS.CODE_TABLE_TYPE,
        .SELECTED_SYMBOLS = 1 << HEADER_OPTIONS.CODE_TABLE_TYPE,

        .CODE_LENGTHS_16  = 0 << HEADER_OPTIONS.CODE_LENGTHS,
        .CODE_LENGTHS_32  = 1 << HEADER_OPTIONS.CODE_LENGTHS,
    };

    struct BaseHeader
    {
        uint32_t width;
        uint32_t blockSize;
        uint8_t headerType;
        uint8_t version;
    };

    static constexpr uint16_t CODE_LENGTHS_HEADER_SIZE{NUMBER_OF_SYMBOLS / 2};
    struct CodeLengthsHeader : public BaseHeader
    {
        uint8_t codeLengths[];
    };

    struct FullHeader : public BaseHeader
    {
        uint8_t buffer[256];
    };

    static constexpr uint16_t ADDITIONAL_HEADER_SIZES[16] = {
        CODE_LENGTHS_HEADER_SIZE,
    };
};

#endif // _COMMON_H_