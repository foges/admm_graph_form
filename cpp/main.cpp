#include <random>
#include <vector>

#include "solver.hpp"

// Non-Negative Least Squares
double test1() {
  size_t m = 10000;
  size_t n = 100000;
  std::vector<double> A(m * n);
  std::vector<double> b(m);
  std::vector<double> x(n);
  std::vector<double> y(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> u_dist(0.0, 1.0);
  std::normal_distribution<double> n_dist(0.0, 1.0);
  for (unsigned int i = 0; i < A.size(); ++i)
    A[i] = 1.0 / static_cast<double>(n) * u_dist(generator);

  AdmmData admm_data;
  admm_data.A = A.data();
  admm_data.x = x.data();
  admm_data.y = y.data();
  admm_data.m = m;
  admm_data.n = n;
  admm_data.rho = 1.0;

  admm_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    admm_data.f.emplace_back(kSquare, 1.0, n_dist(generator) + 1);

  admm_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    admm_data.g.emplace_back(kIndGe0);

  Solver(&admm_data);

  return 0;
}

int main() {
  test1();
}

