#!/bin/bash
N_ITER?=64

SRC = ./src/main.cpp
OUT = ./gpu_flops

$(OUT): $(SRC)
	hipcc -std=c++17 -D N_ITER=$(N_ITER) -w -o $(OUT) $(SRC)

.PHONY: all
all: $(OUT)

clean:
	rm ${OUT}