GSLROOT=/usr/local

CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include -fopenmp
LDFLAGS=-framework Accelerate -lgsl -lgslcblas -lm


# main: main.cpp solver.o
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

#solver.o: solver.cpp solver.hpp prox_lib.hpp
#	$(CXX) $(CXXFLAGS) $< -c -o $@


CUXX=nvcc
CUFLAGS=-arch=sm_20 -lineinfo -G
CULDFLAGS=-L/usr/local/cuda/lib -lcudart -lcublas -L/usr/local/lib

main: main.cpp solver.o solver_link.o
	$(CXX) $(CXXFLAGS) $(CULDFLAGS) $^ -o $@

solver_link.o: solver.o
	$(CUXX) $(CUFLAGS) $< -dlink -o $@

solver.o: solver.cu solver.hpp prox_lib.hpp cml.cuh
	$(CUXX) $(CUFLAGS) $< -dc -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

