# Huffman and RLE Compression of Grayscale Images
This project provides an implementation of a compressor and decompressor intended for grayscale images using Huffman coding and run length encoding. Both the compressor and decompressor are optimized with the use of Intel® Advanced Vector Extensions 512 instructions and allow multi-threaded execution facilitated with OpenMP.

## Directory Structure
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
├── common.h                 - definition of a base class for both the Compressor and Decompressor classes
├── compressor.cpp           - implementation of the Compressor class
├── compressor.h             - definition of the Compressor class
├── decompressor.cpp         - implementation of the Decompressor class
├── decompressor.h           - definition of the Decompressor class
├── kko.proj.zadani24.pdf    - the assignment (in Czech)
├── main.cpp                 - arguments parsing and execution of the specified compression/decompression algorithm
├── main.h                   - definition of the arguments data type and the parsing function 
├── Makefile                 - commands for compilation with various directives
├── README.md
└── report.pdf               - detailed documentation of the project with performance analysis included
```

## Requirements and Compilation
To run the full version of the code, OpenMP compilation and the following instruction sets must be supported:
* `AVX2`,
* `AVX512BW`,
* `AVX512F`,
* `AVX512VL`.

The `Makefile` checks for the availability of these instructions and downgrades the algorithms based on what instructions sets are available. Files compressed with a downgraded compressor yield worse compression rates, but can be decompressed by the full decompressor, vice versa backward compatibility is not, however, supported.

Execute the command `make` to compile the best possible version for your system. If any issues occur, the following alternatives can be tried:
* the command `make no_omp` will compile without OpenMP support and the application will be bound to single threaded execution, 
* the command `make no_avx` will compile with OpenMP support but with fully downgraded algorithms,
* the command `make plain` will compile without OpenMP and with fully downgraded algorithms.

## Usage
```
Usage: ./huff_codec [-c | -d] [-m] [-a] [-w <width>] [-i <input file name>] [-o <output file name>] [-t <number of threads>]
Options:
  -c:                     Compress the input file.
  -d:                     Decompress the input file.
  -m:                     Use the model-based compression (ignored for decompression).
  -a:                     Use the adaptive model-based compression (ignored for decompression).
  -w <width>:             Width of the compressed image (ignored for decompression).
  -i <input file name>:   The input file name.
  -o <output file name>:  The output file names.
  -t <number of threads>: Number of threads to use (default is 4), must be a power of 2 between 1 and 32 inclusive.
```

## Tests
The following commands thoroughly test the implemented algorithms:
* `./bash_src/benchmark_tests.sh` - tests on a benchmark dataset,
* `./bash_src/edge_case_tests.sh` - tests on a dataset with problematic files,
* `./bash_src/random_tests.sh` - randomized tests on files of various sizes and with various data distribution.
