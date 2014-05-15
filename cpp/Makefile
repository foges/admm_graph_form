# User Vars
GSLROOT=/usr/local

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include -fopenmp

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -lineinfo
CULDFLAGS=-lcudart -lcublas

UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS= -lgsl -framework Accelerate -lm
CULDFLAGS_=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS)
else
LDFLAGS=-lgsl -lgslcblas -lm
CULDFLAGS_=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS)
endif

# CPU
cpu: main.cpp solver.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o main

solver.o: solver.cpp solver.hpp prox_lib.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

# GPU
gpu: main.cpp solver_cu.o solver_cu_link.o
	$(CXX) $(CXXFLAGS) $(CULDFLAGS_) $^ -o main

solver_cu_link.o: solver_cu.o
	$(CUXX) $(CUFLAGS) $< -dlink -o $@

solver_cu.o: solver.cu solver.hpp prox_lib.hpp cml.cuh
	$(CUXX) $(CUFLAGS) $< -dc -o $@

# Chol Test
chol_test: chol_test.cu
	nvcc $< $(CUFLAGS) $(CULDFLAGS_) -lgsl -lm -I/usr/local/include -O3 -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

