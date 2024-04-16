MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

# -mavx512vpopcntdq
AVX512 = $(shell grep avx512 /proc/cpuinfo >/dev/null && echo "-mavx512f -mavx512vl -mavx512bw" || echo "")

CC = g++
CFLAGS = -Wall -Wextra -std=c++20 -mavx2 -fopenmp -O3 -Wno-cast-function-type $(AVX512)
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

TARGET = $(PROJECT_DIR)/huff_codec
TARGET_TEST = $(PROJECT_DIR)/huff_codec_test
TARGET_DEBUG = $(PROJECT_DIR)/huff_codec_debug

.PHONY: all test debug clean

all:
	$(CC) $(CFLAGS) $(CPP_SRC) -o $(TARGET)

test:
	$(CC) $(CFLAGS) $(CPP_SRC) -D_TEST_ -o $(TARGET_TEST)

debug:
	$(CC) $(CFLAGS) $(CPP_SRC) -D_DEBUG_ -o $(TARGET_DEBUG)

clean: 
	rm -f $(TARGET) $(TARGET_TEST)
