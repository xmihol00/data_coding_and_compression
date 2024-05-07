# Huffman and RLE Compression of Grayscale Images
This project provides an implementation of a compressor and decompressor intended for grayscale images. Both the compressor and decompressor are optimized with the use of vector AVX instructions and allow multi-threaded execution facilitated with OpenMP.

## Directory structure
```
├── bash_src/                - test, performance analysis and other bash scripts
├── compressed_files/        - should be empty, used for testing 
├── csv_measurements/        - performance measurements
├── data_benchmark/          - set of benchmark images
├── data_edge_case/          - set of test files to verify correct functionality
├── data_random/             - should be empty, used for testing
├── data_traversals/         - set of test files to test adaptive compression
├── decompressed_files/      - should be empty, used for testing
├── pgm_images/              - benchmark images from the data_benchmark/ directory converted to PGM format
├── py_src/                  - python scripts used mostly for performance measurements
├── tex_tables/              - generated LaTeX tables
├── tex_tables_processed/    - adjusted LaTeX tables from the tex_tables/ directory
├── common.h
├── compressor.cpp
├── compressor.h
├── decompressor.cpp
├── decompressor.h
├── kko.proj.zadani24.pdf    - the assignment (in Czech)
├── main.cpp
├── main.h
├── Makefile
├── README.md
└── report.pdf               - detailed documentation of the project with performance analysis included
```