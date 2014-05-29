#include <mex.h>

#include <algorithm>
#include <vector>

#include "solver.hpp"

template <typename T>
void RowToColMajor(const T *Arm, size_t m, size_t n, T *Acm) {
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      Acm[j * m + i] = Arm[i * n + j];
}

template <typename T>
int PopulateFunctionObj(unsigned int n, const mxArray *f_mex,
                        std::vector<FunctionObj<T>> *f_admm) {
  char alpha[] = "f\0a\0b\0c\0d\0";

  int f_idx = mxGetFieldNumber(f_mex, &alpha[0]);
  int a_idx = mxGetFieldNumber(f_mex, &alpha[2]);
  int b_idx = mxGetFieldNumber(f_mex, &alpha[4]);
  int c_idx = mxGetFieldNumber(f_mex, &alpha[6]);
  int d_idx = mxGetFieldNumber(f_mex, &alpha[8]);
  
  if (f_idx == -1) {
      mexErrMsgIdAndTxt("MATLAB:solver:missingParam",
          "Field [f/g].f must be specified.");
      return 1;
  }

  void *f_data = 0, *a_data = 0, *b_data = 0, *c_data = 0, *d_data = 0;
  mxClassID f_id, a_id, b_id, c_id, d_id; 

  mxArray *f_arr = mxGetFieldByNumber(f_mex, 0, f_idx);
  f_data = mxGetPr(f_arr);
  f_id = mxGetClassID(f_arr);
  if (!(mxGetM(f_arr) == n && mxGetN(f_arr) == 1) &&
      !(mxGetN(f_arr) == n && mxGetM(f_arr) == 1)) {
    mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
        "Dimension of [f/g].f and corresponding dimension of A must match.");
    return 1;
  }

  if (a_idx != -1) {
    mxArray *a_arr = mxGetFieldByNumber(f_mex, 0, a_idx);
    a_data = mxGetPr(a_arr);
    a_id = mxGetClassID(a_arr);
    if (!(mxGetM(a_arr) == n && mxGetN(a_arr) == 1) &&
        !(mxGetN(a_arr) == n && mxGetM(a_arr) == 1)) {
      mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
          "Dimension of [f/g].a and [f/g].f must match.");
      return 1;
    }
  }
  if (b_idx != -1) {
    mxArray *b_arr = mxGetFieldByNumber(f_mex, 0, b_idx);
    b_data = mxGetPr(b_arr);
    b_id = mxGetClassID(b_arr);
    if (!(mxGetM(b_arr) == n && mxGetN(b_arr) == 1) &&
        !(mxGetN(b_arr) == n && mxGetM(b_arr) == 1)) {
      mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
          "Dimension of [f/g].b and [f/g].f must match.");
      return 1;
    }
  }
  if (c_idx != -1) {
    mxArray *c_arr = mxGetFieldByNumber(f_mex, 0, c_idx);
    c_data = mxGetPr(c_arr);
    c_id = mxGetClassID(c_arr);
    if (!(mxGetM(c_arr) == n && mxGetN(c_arr) == 1) &&
        !(mxGetN(c_arr) == n && mxGetM(c_arr) == 1)) {
      mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
          "Dimension of [f/g].c and [f/g].f must match.");
      return 1;
    }
  }
  if (d_idx != -1) {
    mxArray *d_arr = mxGetFieldByNumber(f_mex, 0, d_idx);
    d_data = mxGetPr(d_arr);
    d_id = mxGetClassID(d_arr);  
    if (!(mxGetM(d_arr) == n && mxGetN(d_arr) == 1) &&
        !(mxGetN(d_arr) == n && mxGetM(d_arr) == 1)) {
      mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
          "Dimension of [f/g].d and [f/g].f must match.");
      return 1;
    }
  }

  for (unsigned int i = 0; i < n; ++i) {
    T a = static_cast<T>(1);
    T b = static_cast<T>(0);
    T c = static_cast<T>(1);
    T d = static_cast<T>(0);
    Function f;

    if (f_id == mxDOUBLE_CLASS) {
      f = static_cast<Function>(reinterpret_cast<double*>(f_data)[i]);
    } else if (f_id == mxSINGLE_CLASS) {
      f = static_cast<Function>(reinterpret_cast<float*>(f_data)[i]);
    } else if (f_id == mxINT32_CLASS) {
      f = static_cast<Function>(reinterpret_cast<int*>(f_data)[i]);
    } else if (f_id == mxUINT32_CLASS) {
      f = static_cast<Function>(reinterpret_cast<unsigned int*>(f_data)[i]);
    } else if (f_id == mxINT64_CLASS) {
      f = static_cast<Function>(reinterpret_cast<long*>(f_data)[i]);
    } else if (f_id == mxUINT64_CLASS) {
      f = static_cast<Function>(reinterpret_cast<unsigned long*>(f_data)[i]);
    } else {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function type [f/g].f must be double, single, int[32/64], uint[32/64].");
      return 1;
    }

    if (a_data != 0 && a_id == mxDOUBLE_CLASS) {
      a = static_cast<T>(reinterpret_cast<double*>(a_data)[i]);
    } else if (a_data != 0 && a_id ==  mxSINGLE_CLASS) {
      a = static_cast<T>(reinterpret_cast<float*>(a_data)[i]);
    } else if (a_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter [f/g].a must be double or single.");
      return 1;
    }

    if (b_data != 0 && b_id == mxDOUBLE_CLASS) {
      b = static_cast<T>(reinterpret_cast<double*>(b_data)[i]);
    } else if (b_data != 0 && b_id == mxSINGLE_CLASS) {
      b = static_cast<T>(reinterpret_cast<float*>(b_data)[i]);
    } else if (b_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter [f/g].b must be double or single.");
      return 1;
    }

    if (c_data != 0 && c_id == mxDOUBLE_CLASS) {
      c = static_cast<T>(reinterpret_cast<double*>(c_data)[i]);
    } else if (c_data != 0 && c_id == mxSINGLE_CLASS) {
      c = static_cast<T>(reinterpret_cast<float*>(c_data)[i]);
    } else if (c_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter [f/g].c must be double or single.");
      return 1;
    } 

    if (d_data != 0 && d_id == mxDOUBLE_CLASS) {
      d = static_cast<T>(reinterpret_cast<double*>(d_data)[i]);
    } else if (d_data != 0 && d_id == mxSINGLE_CLASS) {
      d = static_cast<T>(reinterpret_cast<float*>(d_data)[i]);
    } else if (d_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter [f/g].d must be double or single.");
      return 1;
    }
    //printf("(%d, %e, %e, %e, %e)\n", f, a, b, c, d);
    f_admm->emplace_back(f, a, b, c, d);
  }
  return 0;
}

template <typename T>
void SolverWrap(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  size_t m = mxGetM(prhs[0]); 
  size_t n = mxGetN(prhs[0]);
  
  T* A = new T[m * n];
  RowToColMajor(reinterpret_cast<T*>(mxGetPr(prhs[0])), n, m, A);

  AdmmData<T, T*> admm_data(A, m, n);
  admm_data.f.reserve(mxGetM(prhs[0]));
  admm_data.g.reserve(mxGetN(prhs[0]));
  admm_data.x = reinterpret_cast<T*>(mxGetPr(plhs[0]));
  if (nlhs == 2)
    admm_data.y = reinterpret_cast<T*>(mxGetPr(plhs[1]));

  int err = PopulateFunctionObj(m, prhs[1], &admm_data.f);
  if (err) return;
  err = PopulateFunctionObj(n, prhs[2], &admm_data.g);
  if (err) return;
  
  Solver(&admm_data);
  delete [] A;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Check arguments.
  if (nrhs < 3 || nrhs > 4) {
    mexErrMsgIdAndTxt("MATLAB:solver:insufficientInputArgs",
        "Usage: [x, y] = solver(A, f, g, [params])");
    return;
  }
  if (nlhs > 2) {
    mexErrMsgIdAndTxt("MATLAB:solver:extraneousOutputArgs",
        "Usage: [x, y] = solver(A, f, g, [params])");
    return;
  }
  
  mxClassID class_id_A = mxGetClassID(prhs[0]);
  mxClassID class_id_f = mxGetClassID(prhs[1]);
  mxClassID class_id_g = mxGetClassID(prhs[2]);

  if (class_id_A != mxSINGLE_CLASS && class_id_A != mxDOUBLE_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
        "Matrix A must either be single or double precision.");
    return;
  }
  if (class_id_f != mxSTRUCT_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:solver:inputNotStruct",
        "Function f must be a struct.");
    return;
  }
  if (class_id_g != mxSTRUCT_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:solver:inputNotStruct",
        "Function g must be a struct.");
    return;
  }
  
  plhs[0] = mxCreateNumericMatrix(mxGetN(prhs[0]), 1, class_id_A, mxREAL);
  if (nlhs == 2)
    plhs[1] = mxCreateNumericMatrix(mxGetM(prhs[0]), 1, class_id_A, mxREAL);

  if (class_id_A == mxDOUBLE_CLASS) {
    SolverWrap<double>(nlhs, plhs, nrhs, prhs);
  } else if (class_id_A == mxSINGLE_CLASS) {
    //SolverWrap<float>(nlhs, plhs, nrhs, prhs);
    mexErrMsgIdAndTxt("MATLAB:solver:inputNotStruct",
        "Function g must be a struct.");

  }
}

