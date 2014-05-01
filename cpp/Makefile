# Instructions to make compilation work
# export PATH=/usr/local/cuda/bin:$PATH
# export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

GSLROOT=/usr/local

CXX=g++
CXXFLAGS=-g -O3 -Wall -std=c++11 -I$(GSLROOT)/include -fopenmp
LDFLAGS=-lgsl -lgslcblas -lm

main: main.cpp solver.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

