#ifndef CUDA_UTIL_CUH_
#define CUDA_UTIL_CUH_

#include <cublas_v2.h>

#include <algorithm>

// Cuda Matrix Library
namespace cml {
// Definitions
typedef unsigned int uint;

const uint kTileSize = 8u; // 32u
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
  cudaMalloc(reinterpret_cast<void**>(&mat.data), m * n * sizeof(T));
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
  cudaFree(A->data);
}

// Submatrix.
template <typename T>
matrix<T> matrix_submatrix(matrix<T> *A, size_t i, size_t j, size_t n1,
                           size_t n2) {
  matrix<T> submat;
  submat.size1 = n1;
  submat.size2 = n2;
  submat.data = A->data + j *A->tda + i;
  submat.tda = A->tda;
  return submat;
}

// Matrix memcpy.
template <typename T>
void matrix_memcpy(matrix<T> *A, const matrix<T> *B) {
  cudaMemcpy(reinterpret_cast<void*>(A->data),
             reinterpret_cast<const void*>(B->data),
             A->tda * A->size2 * sizeof(T), cudaMemcpyDefault);
}

template <typename T>
void matrix_memcpy(matrix<T> *A, const T *B) {
  cudaMemcpy(reinterpret_cast<void*>(A->data), reinterpret_cast<const void*>(B),
             A->tda * A->size2 * sizeof(T), cudaMemcpyDefault);
}

template <typename T>
void matrix_memcpy(T *A, const matrix<T> *B) {
  cudaMemcpy(reinterpret_cast<void*>(A), reinterpret_cast<const void*>(B->data),
             B->tda * B->size2 * sizeof(T), cudaMemcpyDefault);
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
  cudaMalloc(reinterpret_cast<void**>(&vec.data), n * sizeof(T));
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
  cudaFree(x->data);
}

// Subvector.
template <typename T>
vector<T> vector_subvector(vector<T> *vec, size_t offset,
    size_t n) {
  vector<T> subvec;
  subvec.size = n;
  subvec.data = vec->data + offset;
  return subvec;
}

// Vector memcpy.
template <typename T>
void vector_memcpy(vector<T> *x, const vector<T> *y) {
  cudaMemcpy(reinterpret_cast<void*>(x->data),
             reinterpret_cast<const void*>(y->data), x->size * sizeof(T),
             cudaMemcpyDefault);
}

template <typename T>
void vector_memcpy(vector<T> *x, const T *y) {
  cudaMemcpy(reinterpret_cast<void*>(x->data), reinterpret_cast<const void*>(y),
             x->size * sizeof(T), cudaMemcpyDefault);
}

template <typename T>
void vector_memcpy(T *x, const vector<T> *y) {
  cudaMemcpy(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(y->data),
             y->size * sizeof(T), cudaMemcpyDefault);
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
  return cublasSsyrk(handle, uplo, trans, static_cast<int>(C->size1),
                     k, &alpha, A->data, static_cast<int>(A->tda), &beta,
                     C->data, static_cast<int>(C->tda));
}

template <>
cublasStatus_t blas_syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, const double alpha,
                             const matrix<double> *A, const double beta,
                             matrix<double> *C) {
  int k = trans == CUBLAS_OP_N ? A->size2 : A->size1;
  return cublasDsyrk(handle, uplo, trans, static_cast<int>(C->size1), 
                     k, &alpha, A->data, static_cast<int>(A->tda), &beta,
                     C->data, static_cast<int>(C->tda));
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
                             const matrix<double> *B,
                             const matrix<double> *C) {
 return cublasDgeam(handle, transa, transb, static_cast<int>(C->size1),
                    static_cast<int>(C->size2), alpha, A->data,
                    static_cast<int>(A->tda), beta, B->data,
                    static_cast<int>(B->tda), C->data,
                    static_cast<int>(C->tda));
}

template <>
cublasStatus_t blas_geam(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, const float *alpha,
                             const matrix<float> *A, const float *beta,
                             const matrix<float> *B,
                             const matrix<float> *C) {
 return cublasSgeam(handle, transa, transb, static_cast<int>(C->size1),
                    static_cast<int>(C->size2), alpha, A->data,
                    static_cast<int>(A->tda), beta, B->data,
                    static_cast<int>(B->tda), C->data,
                    static_cast<int>(C->tda));
}

// Axpy.
template <typename T>
cublasStatus_t blas_axpy(cublasHandle_t handle, T alpha, vector<T> *x,
                             vector<T> *y);

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, double alpha,
                             vector<double> *x, vector<double> *y) {
  return cublasDaxpy(handle, static_cast<int>(x->size), &alpha, x->data,
                     static_cast<int>(x->stride), y->data,
                     static_cast<int>(y->stride));
}

template <>
cublasStatus_t blas_axpy(cublasHandle_t handle, float alpha,
                             vector<float> *x, vector<float> *y) {
  return cublasSaxpy(handle, static_cast<int>(x->size), &alpha, x->data,
                     static_cast<int>(x->stride), y->data,
                     static_cast<int>(y->stride));
}

// Gemv.
template <typename T>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                             T alpha, matrix<T> *A, const vector<T> *x,
                             T beta, vector<T> *y);

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                             double alpha, matrix<double> *A,
                             const vector<double> *x, double beta,
                             vector<double> *y) {
  return cublasDgemv(handle, trans, static_cast<int>(A->size1),
                     static_cast<int>(A->size2), &alpha, A->data,
                     static_cast<int>(A->tda), x->data,
                     static_cast<int>(x->stride), &beta, y->data,
                     static_cast<int>(y->stride));
}

template <>
cublasStatus_t blas_gemv(cublasHandle_t handle, cublasOperation_t trans,
                             float alpha, matrix<float> *A,
                             const vector<float> *x, float beta,
                             vector<float> *y) {
  return cublasSgemv(handle, trans, static_cast<int>(A->size1),
                     static_cast<int>(A->size2), &alpha, A->data,
                     static_cast<int>(A->tda), x->data,
                     static_cast<int>(x->stride), &beta, y->data,
                     static_cast<int>(y->stride));
}

// Nrm2.
template <typename T>
T blas_nrm2(cublasHandle_t handle, vector<T> *x);

template <>
double blas_nrm2(cublasHandle_t handle, vector<double> *x) {
  double result;
  cublasDnrm2(handle, static_cast<int>(x->size), x->data,
              static_cast<int>(x->stride), &result);
  return result;
}

template <>
float blas_nrm2(cublasHandle_t handle, vector<float> *x) {
  float result;
  cublasSnrm2(handle, static_cast<int>(x->size), x->data,
              static_cast<int>(x->stride), &result);
  return result;
}

// Trsv.
template <typename T>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         matrix<T> *A, vector<T> *x);

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         matrix<double> *A, vector<double> *x) {
  return cublasDtrsv(handle, uplo, trans, diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
}

template <>
cublasStatus_t blas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,
                         cublasOperation_t trans, cublasDiagType_t diag,
                         matrix<float> *A, vector<float> *x) {
  return cublasStrsv(handle, uplo, trans, diag, static_cast<int>(A->size1),
      A->data, static_cast<int>(A->tda), x->data, static_cast<int>(x->stride));
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
  uint num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;
  for (uint i = 0; i < num_tiles; ++i) {
    uint block_dim_1d = std::min<uint>(kTileSize, A->size1 - i * kTileSize);
    dim3 block_dim(block_dim_1d, block_dim_1d);
    block_chol<<<1, block_dim>>>(A->data, i, A->tda);

    if (i < num_tiles - 1u) {
      uint grid_dim = num_tiles - i - 1u;
      block_trsv<<<grid_dim, kTileSize>>>(A->data, i, A->size1, A->tda);

      matrix<T> L12 = matrix_submatrix(A, (i + 1) * kTileSize, i * kTileSize,
          A->size1 - (i + 1) * kTileSize, kTileSize);
      matrix<T> L22 = matrix_submatrix(A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);
      blas_syrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, static_cast<T>(-1),
                &L12, static_cast<T>(1), &L22);
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}

template <typename T>
cublasStatus_t linalg_cholesky_svx(cublasHandle_t handle,
                                   matrix<T> *L, vector<T> *x) {
  
  blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            L, x);

  blas_trsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            L, x); 

  return CUBLAS_STATUS_SUCCESS;
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

