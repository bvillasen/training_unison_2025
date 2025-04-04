#!/bin/bash
amdclang++ -fopenmp --offload-arch=gfx90a -o daxpy src/daxpy.cpp
