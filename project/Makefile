MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

AVX2 = $(shell grep avx2 /proc/cpuinfo >/dev/null && echo "-mavx2" || echo "")
AVX512F = $(shell grep avx512f /proc/cpuinfo >/dev/null && echo "-mavx512f" || echo "") 
AVX512VL = $(shell grep avx512vl /proc/cpuinfo >/dev/null && echo "-mavx512vl" || echo "")
AVX512BW = $(shell grep avx512bw /proc/cpuinfo >/dev/null && echo "-mavx512bw" || echo "")
AVX512VPOPCNTDQ = $(shell grep avx512vpopcntdq /proc/cpuinfo >/dev/null && echo "-mavx512vpopcntdq" || echo "")
AVX512FLAGS = $(AVX2) $(AVX512F) $(AVX512VL) $(AVX512BW) $(AVX512VPOPCNTDQ)

CC = g++
CFLAGS = -Wall -Wextra -std=c++20 -O3 -g
OMPFLAGS = -fopenmp
PRGAMA_IGNORE = -Wno-unknown-pragmas

CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

TARGET = $(PROJECT_DIR)/huff_codec

ADDITIONAL_DEFINES =  # add by running make ADDITIONAL_DEFINES="-D<define name>"

.PHONY: all plain no_omp no_avx debug clean

all:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

plain:
	$(CC) $(CFLAGS) $(CPP_SRC) $(PRGAMA_IGNORE) $(ADDITIONAL_DEFINES) -o $(TARGET)

no_omp:
	$(CC) $(CFLAGS) $(AVX512FLAGS) $(PRGAMA_IGNORE) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

no_avx:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPP_SRC) $(ADDITIONAL_DEFINES) -o $(TARGET)

debug:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(AVX512FLAGS) $(CPP_SRC) -D_DEBUG_PRINT_ACTIVE_ $(ADDITIONAL_DEFINES) -o $(TARGET)

clean: 
	rm -f $(TARGET) $(TARGET_TEST)
