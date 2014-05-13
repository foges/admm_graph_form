#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <vector>

#include "prox_lib.hpp"

// Data structure for input to Solver().
template <typename T>
struct AdmmData {
  // Input.
  std::vector<FunctionObj<T> > f, g;
  const T *A;
  size_t m, n;

  // Output.
  T *x, *y;

  // Parameters.
  T rho;
  unsigned int max_iter;
  T rel_tol, abs_tol;
  bool quiet;

  // Constructor.
  AdmmData(const T *A, size_t m, size_t n)
      : A(A), m(m), n(n), rho(static_cast<T>(1)), max_iter(1000),
        rel_tol(static_cast<T>(1e-2)), abs_tol(static_cast<T>(1e-4)),
        quiet(false) { }
};

template <typename T>
void Solver(AdmmData<T> *admm_data);

#endif /* SOLVER_HPP_ */

