# User Vars
GSLROOT=/usr/local

# C++ Flags
CXX=g++
CXXFLAGS=-g -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include -fopenmp
LDFLAGS=-lgsl -lgslcblas -lm

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -lineinfo
CULDFLAGS=-lcudart -lcublas

UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS_=$(LDFLAGS) -framework Accelerate
CULDFLAGS_=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS)
else
CULDFLAGS_=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS)
endif

# CPU
cpu: main.cpp solver.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS_) $^ -o main

solver.o: solver.cpp solver.hpp prox_lib.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

# GPU
gpu: main.cpp solver_cu.o solver_cu_link.o
	$(CXX) $(CXXFLAGS) $(CULDFLAGS_) $^ -o main

solver_cu_link.o: solver_cu.o
	$(CUXX) $(CUFLAGS) $< -dlink -o $@

solver_cu.o: solver.cu solver.hpp prox_lib.hpp cml.cuh
	$(CUXX) $(CUFLAGS) $< -dc -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

