#include <random>
#include <iostream>

#include "solver.hpp"

// Non-Negative Least Squares
double test1() {
  size_t n = 100;
  size_t m = 10000;
  std::vector<double> A(m * n);
  std::vector<double> b(m);
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
    b[i] = distribution(generator);
    admm_data.f.emplace_back(kSquare, 1.0, b[i], 1.0);
  }

  admm_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    admm_data.g.emplace_back(kIndGe0, 0.0, 0.0, 0.0);
  }

  double err = 0;
  for (int i = 0; i < m; i++) {
    double sum = 0;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * admm_data.x[j];
    }
    err += std::abs(b[i] - sum);
  }
  std::cout << err << std::endl;

  Solver(admm_data);

  err = 0;
  for (int i = 0; i < m; i++) {
    double sum = 0;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * admm_data.x[j];
    }
    err += std::abs(b[i] - sum);
  }
  std::cout << err << std::endl;
  return 1;
}

int main() {
  test1();
}

