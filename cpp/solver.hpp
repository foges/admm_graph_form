#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <vector>

#include "prox_lib.hpp"

// Data structure for input to Solver().
struct AdmmData {
  // Input.
  std::vector<FunctionObj<double>> f;
  std::vector<FunctionObj<double>> g;
  const double *A;
  size_t m, n;

  // Output.
  double *x, *y;

  // Parameters.
  double rho;
  unsigned int max_iter;
  double rel_tol;
  double abs_tol;
  bool quiet;

  // Constructor.
  AdmmData(const double *A, size_t m, size_t n)
      : A(A), m(m), n(n), rho(1.0), max_iter(1000), rel_tol(1e-2),
        abs_tol(1e-4), quiet(false) { }
};

void Solver(AdmmData *admm_data);

#endif /* SOLVER_HPP_ */

