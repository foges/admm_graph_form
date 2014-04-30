#include <random>

#include "solver.hpp"

double test1() {
  size_t n = 10;
  size_t m = 100;
  std::vector<double> A(m * n);
  std::vector<double> x(n);
  std::vector<double> y(m);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (unsigned int i = 0; i < A.size(); ++i)
    A[i] = distribution(generator);

  AdmmData admm_data;
  admm_data.A = A.data();
  admm_data.x = x.data();
  admm_data.y = y.data();
  admm_data.m = m;
  admm_data.n = n;

  admm_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i) {
    double b = distribution(generator);
    admm_data.f.emplace_back(kSquare, 1.0, b, 1.0);
  }

  admm_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    admm_data.g.emplace_back(kZero, 0.0, 0.0, 0.0);
  }

  Solver(admm_data);
  return 1;
}

int main() {
  test1();
}

