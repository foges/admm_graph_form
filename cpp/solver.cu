#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>

#include "cml.cuh"
#include "solver.hpp"

template <typename T> 
void RowToColMajor(const T *Arm, size_t m, size_t n, T *Acm);

template <typename T>
void Solver(AdmmData<T> *admm_data) {
  // Extract values from admm_data
  size_t n = admm_data->n;
  size_t m = admm_data->m;
  bool is_skinny = m >= n;
  size_t min_dim = std::min(m, n);

  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);

  // Create cuBLAS handle.
  cublasHandle_t cb_handle;
  cublasCreate(&cb_handle);

  // Allocate data for ADMM variables.
  cml::vector<T> z = cml::vector_calloc<T>(m + n);
  cml::vector<T> zt = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
  cml::vector<T> z_prev = cml::vector_calloc<T>(m + n);
  cml::matrix<T> L = cml::matrix_alloc<T>(min_dim, min_dim);
  cml::matrix<T> AA = cml::matrix_alloc<T>(min_dim, min_dim);
  cml::matrix<T> A = cml::matrix_alloc<T>(m, n);

  // Copy A to device (assume input row-major).
  T *Acm = new T[m * n];
  RowToColMajor(admm_data->A, m, n, Acm);
  cml::matrix_memcpy(&A, Acm);
  delete [] Acm;

  // Copy f and g to device
  thrust::device_vector<FunctionObj<T> > f(admm_data->f.begin(),
                                           admm_data->f.end());
  thrust::device_vector<FunctionObj<T> > g(admm_data->g.begin(),
                                           admm_data->g.end());
  // Create views for x and y components.
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> xt = cml::vector_subvector(&zt, 0, n);
  cml::vector<T> yt = cml::vector_subvector(&zt, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);

  // Compute cholesky decomposition of (I + A^TA) or (I + AA^T)
  cublasOperation_t mult_type = is_skinny ? CUBLAS_OP_T : CUBLAS_OP_N;
  cml::blas_syrk(cb_handle, CUBLAS_FILL_MODE_LOWER, mult_type, kOne, &A, kZero,
                 &AA);
  cml::matrix_memcpy(&L, &AA);
  cml::matrix_add_constant_diag(&L, kOne);
  cml::linalg_cholesky_decomp(cb_handle, &L);

  cml::print_matrix(AA);
  cml::print_matrix(L);

  // Signal start of execution.
  if (!admm_data->quiet)
    printf("%4s %12s %10s %10s %10s %10s\n",
           "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

  T sqrtn_atol = sqrt(static_cast<T>(n)) * admm_data->abs_tol;

  for (unsigned int k = 0; k < admm_data->max_iter; ++k) {
    // Evaluate Proximal Operators
    cml::blas_axpy(cb_handle, -kOne, &xt, &x);
    cml::blas_axpy(cb_handle, -kOne, &yt, &y);

    cml::print_vector(x);
    cml::print_vector(y);

    ProxEval(g, admm_data->rho, x.data, x12.data);
    ProxEval(f, admm_data->rho, y.data, y12.data);
    
    cml::print_vector(x12);
    cml::print_vector(y12);
    exit(1);

    // Project and Update Dual Variables
    cml::blas_axpy(cb_handle, kOne, &x12, &xt);
    cml::blas_axpy(cb_handle, kOne, &y12, &yt);
    if (is_skinny) {
      cml::vector_memcpy(&x, &xt);
      cml::blas_gemv(cb_handle, CUBLAS_OP_T, kOne, &A, &yt, kOne, &x);
      cml::linalg_cholesky_svx(cb_handle, &L, &x);
      cml::blas_gemv(cb_handle, CUBLAS_OP_N, kOne, &A, &x, kZero, &y);
      cml::blas_axpy(cb_handle, -kOne, &y, &yt);
    } else {
      cml::blas_gemv(cb_handle, CUBLAS_OP_N, kOne, &A, &xt, kZero, &y);
      cml::blas_symv(cb_handle, CUBLAS_FILL_MODE_LOWER, kOne, &AA, &yt, kOne,
                     &y);
      cml::linalg_cholesky_svx(cb_handle, &L, &y);
      cml::blas_axpy(cb_handle, -kOne, &y, &yt);
      cml::vector_memcpy(&x, &xt);
      cml::blas_gemv(cb_handle, CUBLAS_OP_T, kOne, &A, &yt, kOne, &x);
    }
    cml::blas_axpy(cb_handle, -kOne, &x, &xt);

    // Compute primal and dual tolerances.
    T nrm_z = cml::blas_nrm2(cb_handle, &z);
    T nrm_zt = cml::blas_nrm2(cb_handle, &zt);
    T nrm_z12 = cml::blas_nrm2(cb_handle, &z12);
    T eps_pri = sqrtn_atol + admm_data->rel_tol * std::max(nrm_z12, nrm_z);
    T eps_dual = sqrtn_atol + admm_data->rel_tol * admm_data->rho * nrm_zt;

    // Compute ||r^k||_2 and ||s^k||_2.
    cml::blas_axpy(cb_handle, -kOne, &z, &z12);
    cml::blas_axpy(cb_handle, -kOne, &z, &z_prev);
    T nrm_r = cml::blas_nrm2(cb_handle, &z12);
    T nrm_s = admm_data->rho * cml::blas_nrm2(cb_handle, &z_prev);

    // Evaluate stopping criteria.
    bool converged = nrm_r <= eps_pri && nrm_s <= eps_dual;
    if (!admm_data->quiet && (k % 10 == 0 || converged)) {
      T obj = FuncEval(f, y.data) + FuncEval(g, x.data);
      printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e\n",
             k, nrm_r, eps_pri, nrm_s, eps_dual, obj);
    }

    if (converged)
      break;

    // Make copy of z.
    cml::vector_memcpy(&z_prev, &z);
  }
  
  cml::vector_memcpy(admm_data->y, &y12);
  cml::vector_memcpy(admm_data->x, &x12);

  // Free up memory.
  cml::matrix_free(&L);
  cml::matrix_free(&AA);
  cml::matrix_free(&A);
  cml::vector_free(&z);
  cml::vector_free(&zt);
  cml::vector_free(&z12);
  cml::vector_free(&z_prev);
}

template <typename T> 
void RowToColMajor(const T *Arm, size_t m, size_t n, T *Acm) {
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      Acm[j * m + i] = Arm[i * n + j];
}

template void Solver<double>(AdmmData<double> *);
template void Solver<float>(AdmmData<float> *);

