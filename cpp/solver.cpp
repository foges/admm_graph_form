#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <algorithm>
#include <vector>

#include "timer.hpp"
#include "solver.hpp"

#ifdef __MEX__
#define printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);
#endif  // __MEX__

template<>
void Solver(AdmmData<double, double*> *admm_data) {
  // Extract values from admm_data
  size_t n = admm_data->n;
  size_t m = admm_data->m;
  bool is_skinny = m >= n;
  size_t min_dim = std::min(m, n);

  gsl_matrix_const_view A = gsl_matrix_const_view_array(admm_data->A, m, n);

  // Allocate data for ADMM variables.
  gsl_vector *z = gsl_vector_calloc(m + n);
  gsl_vector *zt = gsl_vector_calloc(m + n);
  gsl_vector *z12 = gsl_vector_calloc(m + n);
  gsl_vector *z_prev = gsl_vector_calloc(m + n);
  gsl_matrix *L = gsl_matrix_calloc(min_dim, min_dim);
  gsl_matrix *AA = gsl_matrix_calloc(min_dim, min_dim);

  // Create views for x and y components.
  gsl_vector_view x = gsl_vector_subvector(z, 0, n);
  gsl_vector_view y = gsl_vector_subvector(z, n, m);
  gsl_vector_view xt = gsl_vector_subvector(zt, 0, n);
  gsl_vector_view yt = gsl_vector_subvector(zt, n, m);
  gsl_vector_view x12 = gsl_vector_subvector(z12, 0, n);
  gsl_vector_view y12 = gsl_vector_subvector(z12, n, m);

  // Compute cholesky decomposition of (I + A^TA) or (I + AA^T)
  CBLAS_TRANSPOSE_t mult_type = is_skinny ? CblasTrans : CblasNoTrans;
  gsl_blas_dsyrk(CblasLower, mult_type, 1.0, &A.matrix, 0.0, AA);
  gsl_matrix_memcpy(L, AA);
  for (unsigned int i = 0; i < min_dim; ++i)
    *gsl_matrix_ptr(L, i, i) += 1.0;
  gsl_linalg_cholesky_decomp(L);

  // Signal start of execution.
  if (!admm_data->quiet)
    printf("%4s %12s %10s %10s %10s %10s\n",
           "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

  double sqrtn_atol = sqrt(static_cast<double>(n)) * admm_data->abs_tol;

  for (unsigned int k = 0; k < admm_data->max_iter; ++k) {
    // Evaluate Proximal Operators
    gsl_vector_sub(&x.vector, &xt.vector);
    gsl_vector_sub(&y.vector, &yt.vector);
    ProxEval(admm_data->g, admm_data->rho, x.vector.data, x12.vector.data);
    ProxEval(admm_data->f, admm_data->rho, y.vector.data, y12.vector.data);

    // Project and Update Dual Variables
    gsl_vector_add(&xt.vector, &x12.vector);
    gsl_vector_add(&yt.vector, &y12.vector);
    if (is_skinny) {
      gsl_vector_memcpy(&x.vector, &xt.vector);
      gsl_blas_dgemv(CblasTrans, 1.0, &A.matrix, &yt.vector, 1.0, &x.vector);
      gsl_linalg_cholesky_svx(L, &x.vector);
      gsl_blas_dgemv(CblasNoTrans, 1.0, &A.matrix, &x.vector, 0.0, &y.vector);
      gsl_vector_sub(&yt.vector, &y.vector);
    } else {
      gsl_blas_dgemv(CblasNoTrans, 1.0, &A.matrix, &xt.vector, 0.0, &y.vector);
      gsl_blas_dsymv(CblasLower, 1.0, AA, &yt.vector, 1.0, &y.vector);
      gsl_linalg_cholesky_svx(L, &y.vector);
      gsl_vector_sub(&yt.vector, &y.vector);
      gsl_vector_memcpy(&x.vector, &xt.vector);
      gsl_blas_dgemv(CblasTrans, 1.0, &A.matrix, &yt.vector, 1.0, &x.vector);
    }
    gsl_vector_sub(&xt.vector, &x.vector);

    // Compute primal and dual tolerances.
    double nrm_z = gsl_blas_dnrm2(z);
    double nrm_zt = gsl_blas_dnrm2(zt);
    double nrm_z12 = gsl_blas_dnrm2(z12);
    double eps_pri = sqrtn_atol + admm_data->rel_tol * std::max(nrm_z12, nrm_z);
    double eps_dual = sqrtn_atol + admm_data->rel_tol * admm_data->rho * nrm_zt;

    // Compute ||r^k||_2 and ||s^k||_2.
    gsl_vector_sub(z12, z);
    gsl_vector_sub(z_prev, z);
    double nrm_r = gsl_blas_dnrm2(z12);
    double nrm_s = admm_data->rho * gsl_blas_dnrm2(z_prev);

    // Evaluate stopping criteria.
    bool converged = nrm_r <= eps_pri && nrm_s <= eps_dual;
    if (!admm_data->quiet && (k % 10 == 0 || converged)) {
      double obj = FuncEval(admm_data->f, y.vector.data) +
          FuncEval(admm_data->g, x.vector.data);
      printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, obj);
    }

    if (converged)
      break;

    // Make copy of z.
    gsl_vector_memcpy(z_prev, z);
  }

  // Copy results to output.
  for (unsigned int i = 0; i < m && admm_data->y != 0; ++i)
    admm_data->y[i] = gsl_vector_get(&y.vector, i);
  for (unsigned int i = 0; i < n && admm_data->x != 0; ++i)
    admm_data->x[i] = gsl_vector_get(&x.vector, i);

  // Free up memory.
  gsl_matrix_free(L);
  gsl_matrix_free(AA);
  gsl_vector_free(z);
  gsl_vector_free(zt);
  gsl_vector_free(z12);
  gsl_vector_free(z_prev);
}

