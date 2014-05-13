#ifndef UTILS_H__
#define UTILS_H__

#include <cublas_v2.h>
#include <cstdio>

#define CudaCheckError(val) __CudaCE( (val), __func__, __FILE__, __LINE__)
#define CublasCheckError(val) __CublasCE( (val), __func__, __FILE__, __LINE__)

static const char* cublasGetErrorString(cublasStatus_t error);

template<typename T>
void __CudaCE(T err, const char* const func, const char* const file,
              const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file, line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
  }
}

template<typename T>
void __CublasCE(T err, const char* const func, const char* const file,
                const int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error at: %s : %d\n", file, line);
    fprintf(stderr, "%s %s\n", cublasGetErrorString(err), func);
  }
}

static const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "<unknown>";
  }
}

#endif
