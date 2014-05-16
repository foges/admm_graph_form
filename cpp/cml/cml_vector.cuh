#ifndef CML_VECTOR_CUH_
#define CML_VECTOR_CUH_

#include <cstdio>

#include "cml_defs.cuh"
#include "cml_utils.cuh"

// Cuda Matrix Library
namespace cml {

// Vector Class
template <typename T>
struct vector {
  size_t size, stride;
  T* data;
};

// Helper methods
namespace {

template <typename T>
__global__ void __set_vector(T *data, T val, size_t stride, size_t size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

template <typename T>
void _set_vector(vector<T> *x, T val) {
  uint grid_dim = calc_grid_dim(x->size, kBlockSize);
  __set_vector<<<grid_dim, kBlockSize>>>(x->data, val, x->stride, x->size);
}

}  // namespace

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
  _set_vector(&vec, static_cast<T>(0));
  return vec;
}

template<typename T>
void vector_free(vector<T> *x) {
  cudaError_t err = cudaFree(x->data);
  CudaCheckError(err);
}

template <typename T>
vector<T> vector_subvector(vector<T> *vec, size_t offset, size_t n) {
  vector<T> subvec;
  subvec.size = n;
  subvec.data = vec->data + offset * vec->stride;
  subvec.stride = vec->stride;
  return subvec;
}

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

}  // namespace cml

#endif  // CML_VECTOR_CUH_

