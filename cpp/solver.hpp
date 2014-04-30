#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "prox_lib.hpp"

struct AdmmData {
  double rho;
  std::vector<FunctionObj<double>> f;
  std::vector<FunctionObj<double>> g;
  double *A, *x, *y;
  size_t m, n;
};

void Solver(AdmmData &admm_data) {
  const unsigned int kMaxIter = 100;
  const double kRelTol = 1e-2;
  const double kAbsTol = 1e-4;

  /* Extract Values from admm_data */
  size_t n = admm_data.n;
  size_t m = admm_data.m;
  gsl_matrix_view A = gsl_matrix_view_array(admm_data.A, m, n);
  
  bool is_skinny = m >= n;
  size_t min_dim  = std::min(m, n);

  // Allocate data for ADMM variables
  gsl_vector *x = gsl_vector_calloc(n);
  gsl_vector *y = gsl_vector_calloc(m);
  gsl_vector *xt = gsl_vector_calloc(n);
  gsl_vector *yt = gsl_vector_calloc(m);
  gsl_vector *x12 = gsl_vector_calloc(n);
  gsl_vector *y12 = gsl_vector_calloc(m);

  gsl_matrix *L = gsl_matrix_calloc(min_dim, min_dim);
  gsl_matrix *AA = gsl_matrix_calloc(min_dim, min_dim);
  gsl_matrix *I = gsl_matrix_calloc(min_dim, min_dim);

  // Compute Cholesky Decomposition of (I + AA^T) or (I + A^TA)
  CBLAS_TRANSPOSE_t mult_type = is_skinny ? CblasTrans : CblasNoTrans;
  gsl_blas_dsyrk(CblasLower, mult_type, 1.0, &A.matrix, 0.0, AA);
  gsl_matrix_set_identity(I);
  gsl_matrix_memcpy(L, AA);
  gsl_matrix_add(L, I);
  gsl_linalg_cholesky_decomp(L);
  gsl_matrix_free(I);

  for (unsigned int i = 0; i < kMaxIter; ++i) {
    // Evaluate Proximal Operators
    gsl_vector_sub(x, xt);
    gsl_vector_sub(y, yt);
    ProxEval(admm_data.g, admm_data.rho, x->data, x12->data); 
    ProxEval(admm_data.f, admm_data.rho, y->data, y12->data); 
    
    // Project and Update Dual Variables
    gsl_vector_add(xt, x12);
    gsl_vector_add(yt, y12);
    if (is_skinny) {
      gsl_vector_memcpy(x, xt);
      gsl_blas_dgemv(CblasTrans, 1.0, &A.matrix, yt, 1.0, x);
      gsl_linalg_cholesky_solve(L, x, x);
      gsl_blas_dgemv(CblasNoTrans, 1.0, &A.matrix, x, 0, y);
      gsl_vector_sub(yt, y);
    } else {
      gsl_blas_dgemv(CblasNoTrans, 1.0, &A.matrix, xt, 0.0, y);
      gsl_blas_dgemv(CblasNoTrans, 1.0, AA, yt, 1.0, y);
      gsl_linalg_cholesky_solve(L, y, y);
      gsl_vector_sub(yt, y);
      gsl_vector_memcpy(x, xt);
      gsl_blas_dgemv(CblasTrans, 1.0, &A.matrix, yt, 1.0, x);
    }
    gsl_vector_sub(xt, x);
  }

  for (unsigned int i = 0; i < m; ++i)
    admm_data.y[i] = gsl_vector_get(y, i);

  for (unsigned int i = 0; i < n; ++i)
    admm_data.x[i] = gsl_vector_get(x, i);

  gsl_matrix_free(L);
  gsl_matrix_free(AA);
  gsl_vector_free(x);
  gsl_vector_free(y);
  gsl_vector_free(xt);
  gsl_vector_free(yt);
  gsl_vector_free(x12);
  gsl_vector_free(y12);
}


#endif /* SOLVER_HPP_ */

