MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

AVX2 = $(shell grep avx2 /proc/cpuinfo >/dev/null && echo "-mavx2" || echo "")
AVX512F = $(shell grep avx512f /proc/cpuinfo >/dev/null && echo "-mavx512f" || echo "") 
AVX512VL = $(shell grep avx512vl /proc/cpuinfo >/dev/null && echo "-mavx512vl" || echo "")
AVX512BW = $(shell grep avx512bw /proc/cpuinfo >/dev/null && echo "-mavx512bw" || echo "")
AVX512VPOPCNTDQ = $(shell grep avx512vpopcntdq /proc/cpuinfo >/dev/null && echo "-mavx512vpopcntdq" || echo "")
AVX512FLAGS = $(AVX2) $(AVX512F) $(AVX512VL) $(AVX512BW) $(AVX512VPOPCNTDQ)

CC = $(shell command -v icpx >/dev/null 2>&1 && echo "icpx" || echo "g++")
CFLAGS = -Wall -Wextra -std=c++20 -O3
OMPFLAGS = -fopenmp
PRGAMA_IGNORE = -Wno-unknown-pragmas

CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

TARGET = $(PROJECT_DIR)/huff_codec
PACK_TARGET= xmihol00.zip

ADDITIONAL_DEFINES =  # add by running make ADDITIONAL_DEFINES="-D<define name>"

.PHONY: all plain no_omp no_avx debug plain_debug clean partial_measure algorithm_measure full_measure data_analysis pack

all:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

plain:
	$(CC) $(CFLAGS) $(CPP_SRC) $(PRGAMA_IGNORE) $(ADDITIONAL_DEFINES) -o $(TARGET)

no_omp:
	$(CC) $(CFLAGS) $(AVX512FLAGS) $(PRGAMA_IGNORE) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

no_avx:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

debug:
	$(CC) $(CFLAGS) -g $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_DEBUG_PRINT_ACTIVE_ $(ADDITIONAL_DEFINES) -o $(TARGET)

plain_debug:
	$(CC) $(CFLAGS) -g $(CPP_SRC) -D_DEBUG_PRINT_ACTIVE_ $(PRGAMA_IGNORE) $(ADDITIONAL_DEFINES) -o $(TARGET)

partial_measure:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_MEASURE_PARTIAL_EXECUTION_TIMES_ $(ADDITIONAL_DEFINES) -o $(TARGET)

algorithm_measure:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_MEASURE_ALGORITHM_EXECUTION_TIMES_ $(ADDITIONAL_DEFINES) -o $(TARGET)

full_measure:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_MEASURE_FULL_EXECUTION_TIME_ $(ADDITIONAL_DEFINES) -o $(TARGET)

data_analysis:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_PERFORM_DATA_ANALYSIS_ $(ADDITIONAL_DEFINES) -o $(TARGET)

clean: 
	rm -f $(TARGET) $(PACK_TARGET)

pack: clean
	zip -r $(PACK_TARGET) ./* -x kko.proj.zadani24.pdf "compressed_files/*" "csv_measurements/*" "decompressed_files/*" "pgm_images/*" "data_random/*"