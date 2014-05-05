GSLROOT=/usr/local

CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include -fopenmp
LDFLAGS=-framework Accelerate -lgsl -lgslcblas -lm

main: main.cpp solver.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

solver.o: solver.cpp solver.hpp prox_lib.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

