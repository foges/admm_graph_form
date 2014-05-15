#ifndef CUDA_UTIL_CUH_
#define CUDA_UTIL_CUH_

#include <cublas_v2.h>

#include <algorithm>
#include <thrust/functional.h>

#include "utils.cuh"

// Cuda Matrix Library
namespace cml {
// Definitions
typedef unsigned int uint;

const uint kTileSize = 32u;
const uint kBlockSize = 256u;
const uint kMaxGridSize = 65535u;

namespace {
// Set element kernel.
template <typename T>
__global__ void set_elems__(T val, uint numel, T *d_dst) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < numel; i += gridDim.x * blockDim.x)
    d_dst[i] = val;
}

template <typename T>
void set_elems(T val, uint numel, T* d_dst) {
  uint grid_dim = std::min((numel + kBlockSize - 1) / kBlockSize, kMaxGridSize);
  set_elems__<<<grid_dim, kBlockSize>>>(val, numel, d_dst);
}

template <typename T>
__global__ void matrix_add_constant_diag__(T *data, T val, size_t tda) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i * tda + i] += val;
}
}  // namespace


//////////////////////////////////////////////////
//////////// Matrix - Column Major ///////////////
//////////////////////////////////////////////////
template <typename T>
struct matrix {
  size_t size1, size2, tda;
  T* data;
};

template <typename T>
matrix<T> matrix_alloc(size_t m, size_t n) {
  matrix<T> mat;
  mat.size1 = m;
  mat.size2 = n;
  mat.tda = m;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&mat.data), 
      m * n * sizeof(T));
  CudaCheckError(err);
  return mat;
}

template <typename T>
matrix<T> matrix_calloc(size_t m, size_t n) {
  matrix<T> mat = matrix_alloc<T>(m, n);
  set_elems(static_cast<T>(0), m * n, mat.data);
  return mat;
}

template<typename T>
void matrix_free(matrix<T> *A) {
  cudaError_t err = cudaFree(A->data);
  CudaCheckError(err);
}

// Submatrix.
template <typename T>
matrix<T> matrix_submatrix(matrix<T> *A, size_t i, size_t j, size_t n1,
                           size_t n2) {
  matrix<T> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  submat.data = A->data + j * A->tda + i;
  submat.tda = A->tda;
  return submat;
}

// Matrix memcpy.
// TODO: Take tda into account properly
template <typename T>
void matrix_memcpy(matrix<T> *A, const matrix<T> *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A->data),
      reinterpret_cast<const void*>(B->data), A->tda * A->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const T *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A->data),
      reinterpret_cast<const void*>(B), A->tda * A->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void matrix_memcpy(T *A, const matrix<T> *B) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(A),
      reinterpret_cast<const void*>(B->data), B->tda * B->size2 * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void print_matrix(const matrix<T> &A) {
  T* A_ = new T[A.tda * A.size2];
  matrix_memcpy(A_, &A);
  for (unsigned int i = 0; i < A.size1; ++i) {
    for (unsigned int j = 0; j < A.size2; ++j)
      printf("%e ", A_[i + j * A.tda]);
    printf("\n");
  }
  printf("\n");
  delete [] A_;
}


//////////////////////////////////////////////////
//////////////////// Vector //////////////////////
//////////////////////////////////////////////////
template <typename T>
struct vector {
  size_t size, stride;
  T* data;
};

template <typename T>
vector<T> vector_alloc(size_t n) {
  vector<T> vec;
  vec.size = n;
  vec.stride = 1;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&vec.data),
      n * sizeof(T));
  return vec;
}

template <typename T>
vector<T> vector_calloc(size_t n) {
  vector<T> vec = vector_alloc<T>(n);
  set_elems(static_cast<T>(0), n, vec.data);
  return vec;
}

template<typename T>
void vector_free(vector<T> *x) {
  cudaError_t err = cudaFree(x->data);
  CudaCheckError(err);
}

// Subvector.
template <typename T>
vector<T> vector_subvector(vector<T> *vec, size_t offset, size_t n) {
  vector<T> subvec;
  subvec.size = n;
  subvec.data = vec->data + offset * vec->stride;
  subvec.stride = vec->stride;
  return subvec;
}

// Vector memcpy.
// TODO: Take stride into account.
template <typename T>
void vector_memcpy(vector<T> *x, const vector<T> *y) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(x->data),
      reinterpret_cast<const void*>(y->data), x->size * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void vector_memcpy(vector<T> *x, const T *y) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(x->data),
      reinterpret_cast<const void*>(y), x->size * sizeof(T), cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void vector_memcpy(T *x, const vector<T> *y) {
  cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(x),
      reinterpret_cast<const void*>(y->data), y->size * sizeof(T),
      cudaMemcpyDefault);
  CudaCheckError(err);
}

template <typename T>
void print_vector(const vector<T> &x) {
  T* x_ = new T[x.size * x.stride];
  vector_memcpy(x_, &x);
  for (unsigned int i = 0; i < x.size; ++i)
    printf("%e ", x_[i * x.stride]);
  printf("\n");
  delete [] x_;
}


//////////////////////////////////////////////////
///////////////////// Math ///////////////////////
//////////////////////////////////////////////////

template <typename T>
__device__ T math_sqrt(T x);

template <>
__device__ double math_sqrt(double x) {
  return sqrt(x);
}

template <>
__device__ float math_sqrt(float x) {
  return sqrtf(x);
}

template <typename T>
__device__ T math_rsqrt(T x);

template <>
__device__ double math_rsqrt(double x) {
  return rsqrt(x);
}

template <>
__device__ float math_rsqrt(float x) {
  return rsqrtf(x);
}

//////////////////////////////////////////////////
///////////////////// BLAS ///////////////////////
//////////////////////////////////////////////////

// Syrk.
template <typename T>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const T alpha,
                         const matrix<T> *A, const T beta,
                         matrix<T> *C);

template <>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, const float alpha,
                         const matrix<float> *A, const float beta,
                         matrix<float> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err = cublasSsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;

}

template <>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, const double alpha,
                             const matrix<double> *A, const double beta,
                             matrix<double> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  cublasStatus_t err = cublasDsyrk(handle, uplo, trans,
      static_cast<int>(C->size1), k, &alpha, A->data, static_cast<int>(A->tda),
      &beta, C->data, static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;

}

// Geam.
template <typename T>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const T *alpha,
                         const matrix<T> *A, const T *beta,
                         const matrix<T> *B, const matrix<T> *C);

template <>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const double *alpha,
                         const matrix<double> *A, const double *beta,
                         const matrix<double> *B, const matrix<double> *C) {
 cublasStatus_t err = cublasDgeam(handle, transa, transb,
     static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
     static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda), C->data,
     static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, const float *alpha,
                         const matrix<float> *A, const float *beta,
                         const matrix<float> *B, const matrix<float> *C) {
 cublasStatus_t err = cublasSgeam(handle, transa, transb,
     static_cast<int>(C->size1), static_cast<int>(C->size2), alpha, A->data,
     static_cast<int>(A->tda), beta, B->data, static_cast<int>(B->tda), C->data,
     static_cast<int>(C->tda));
  CublasCheckError(err);
  return err;
}

// Axpy.
template <typename T>
cublasStatus_t blas_axpy(cublasHandle_t handle, T alpha, const vector<T> *x,
                         vector<T> *y);

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, double alpha,
                         const vector<double> *x, vector<double> *y) {
  cublasStatus_t err = cublasDaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, float alpha,
                         const vector<float> *x, vector<float> *y) {
  cublasStatus_t err = cublasSaxpy(handle, static_cast<int>(x->size), &alpha,
      x->data, static_cast<int>(x->stride), y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Gemv.
template <typename T>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         T alpha, matrix<T> *A, const vector<T> *x, T beta,
                         vector<T> *y);

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         double alpha, matrix<double> *A,
                         const vector<double> *x, double beta,
                         vector<double> *y) {
  cublasStatus_t err = cublasDgemv(handle, trans, static_cast<int>(A->size1),
      static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), &beta, y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                         float alpha, matrix<float> *A, const vector<float> *x,
                         float beta, vector<float> *y) {
  cublasStatus_t err = cublasSgemv(handle, trans, static_cast<int>(A->size1),
      static_cast<int>(A->size2), &alpha, A->data, static_cast<int>(A->tda),
      x->data, static_cast<int>(x->stride), &beta, y->data,
      static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Symv.
template <typename T>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                        T alpha, matrix<T> *A, const vector<T> *x, T beta, 
                        vector<T> *y);

template <>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                         double alpha, matrix<double> *A,
                         const vector<double> *x, double beta,
                         vector<double> *y) {
  cublasStatus_t err = cublasDsymv(handle, uplo, static_cast<int>(A->size1),
      &alpha, A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_symv(cublasHandle_t handle, cublasFillMode_t uplo,
                         float alpha, matrix<float> *A, const vector<float> *x,
                         float beta, vector<float> *y) {
  cublasStatus_t err = cublasSsymv(handle, uplo, static_cast<int>(A->size1),
      &alpha, A->data, static_cast<int>(A->tda), x->data,
      static_cast<int>(x->stride), &beta, y->data, static_cast<int>(y->stride));
  CublasCheckError(err);
  return err;
}

// Nrm2.
template <typename T>
struct Square : thrust::unary_function<T, T> {
  __device__ T operator()(const T &x) {
    return x * x;
  }
};

template <typename T>
T blas_nrm2(cublasHandle_t handle, vector<T> *x) {
  return sqrt(thrust::transform_reduce(thrust::device_pointer_cast(x->data),
      thrust::device_pointer_cast(x->data + x->size), Square<T>(),
      static_cast<T>(0.0), thrust::plus<T>()));
}

template <typename T>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<T> *x, T *result);

template <>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<double> *x,
                         double *result) {
  cublasStatus_t err = cublasDnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_nrm2(cublasHandle_t handle, vector<float> *x,
                         float *result) {
  cublasStatus_t err = cublasSnrm2(handle, static_cast<int>(x->size), x->data,
      static_cast<int>(x->stride), result);
  CublasCheckError(err);
  return err;
}

// Trsv.
template <typename T>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<T> *A, vector<T> *x);

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<double> *A, vector<double> *x) {
  cublasStatus_t err = cublasDtrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data, 
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         const matrix<float> *A, vector<float> *x) {
  cublasStatus_t err = cublasStrsv(handle, uplo, trans, diag,
      static_cast<int>(A->size1), A->data, static_cast<int>(A->tda), x->data, 
      static_cast<int>(x->stride));
  CublasCheckError(err);
  return err;
}


//////////////////////////////////////////////////
//////////////////// Linalg //////////////////////
//////////////////////////////////////////////////

// Cholesky Helpers.
template <typename T>
__global__ void block_chol(T *A, uint iter, uint tda) {
  uint col = threadIdx.x;
  uint row = threadIdx.y;
  uint mat_dim = blockDim.x;

  uint global_col = iter * kTileSize + col;
  uint global_row = iter * kTileSize + row;

  const uint kSmTda = kTileSize + 1u;

  __shared__ T L[kSmTda * kTileSize];
  L[row + col * kSmTda] = A[global_row + global_col * tda];
  __syncthreads();

  for (uint i = 0; i < mat_dim; ++i) {
    T rl11 = math_rsqrt(L[i + i * kSmTda]);
    __syncthreads();
    if (row >= i && col == 0) 
      L[row + i * kSmTda] *= rl11;
    __syncthreads();
    if (row >= col && col > i)
      L[row + col * kSmTda] -= L[col + i * kSmTda] * L[row + i * kSmTda];
    __syncthreads();
  }

  if (row >= col)
    A[global_row + global_col * tda] = L[row + col * kSmTda];
}

template <typename T>
__global__ void block_trsv(T *A, uint iter, uint n, uint tda) {
  uint tile_idx = blockIdx.x;
  uint row = threadIdx.x;
  
  const uint kSmTda = kTileSize + 1u;
  __shared__ T L[kSmTda * kTileSize];
  __shared__ T A12[kSmTda * kTileSize];

  uint global_col = iter * kTileSize;
  uint global_row = iter * kTileSize + row;
  
  // Load A -> L column-wise.
  for (uint i = 0; i < kTileSize; ++i)
    L[row + i * kSmTda] = A[global_row + (global_col + i) * tda];
  
  global_row = row + (iter + tile_idx + 1u) * kTileSize;
  
  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      A12[row + i * kSmTda] = A[global_row + (global_col + i) * tda];
  }
  __syncthreads();
  
  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i) {
      for (uint j = 0; j < i; ++j)
        A12[row + i * kSmTda] -= A12[row + j * kSmTda] * L[i + j * kSmTda];
      A12[row + i * kSmTda] /= L[i + i * kSmTda];
    }
  }
  __syncthreads();
  
  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      A[global_row + (global_col + i) * tda] = A12[row + i * kSmTda];
  }
}

// Cholesky.
template <typename T>
cublasStatus_t linalg_cholesky_decomp(cublasHandle_t handle,
                                      matrix<T> *A) {
  cudaStream_t stm;
  cublasStatus_t err = cublasGetStream(handle, &stm);

  uint num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;
  for (uint i = 0; i < num_tiles; ++i) {
    if (err != CUBLAS_STATUS_SUCCESS)
      break;
    uint block_dim_1d = std::min<uint>(kTileSize, A->size1 - i * kTileSize);
    dim3 block_dim(block_dim_1d, block_dim_1d);
    block_chol<<<1, block_dim, 0, stm>>>(A->data, i, A->tda);
    if (i < num_tiles - 1u) {
      uint grid_dim = num_tiles - i - 1u;
      block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i, A->size1, A->tda);

      matrix<T> L12 = matrix_submatrix(A, (i + 1) * kTileSize, i * kTileSize,
          A->size1 - (i + 1) * kTileSize, kTileSize);
      matrix<T> L22 = matrix_submatrix(A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);
      err = blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
         static_cast<T>(-1), &L12, static_cast<T>(1), &L22);
    }
  }
  CublasCheckError(err);
  return err;
}

template <typename T>
cublasStatus_t linalg_cholesky_svx(cublasHandle_t handle,
                                   const matrix<T> *L, vector<T> *x) {
  
  cublasStatus_t err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, L, x);
  CublasCheckError(err);
  
  err = blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT, L, x); 
  CublasCheckError(err);

  return err;
}






// Add constant to diagonal.
template <typename T>
cublasStatus_t matrix_add_constant_diag(matrix<T> *A, T val) {
  uint numel = std::min(A->size1, A->size2);
  uint block_size = std::min(kBlockSize, numel);
  uint grid_size = (numel + block_size - 1) / block_size;
  matrix_add_constant_diag__<<<grid_size, block_size>>>(A->data, val,
                                                            A->tda);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace
#endif /* CUDA_UTIL_CUH_ */

