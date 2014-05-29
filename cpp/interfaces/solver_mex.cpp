#include <mex.h>

#include <algorithm>
#include <vector>

#include "solver.hpp"

// Converts Row -> Column Major
// TODO: Do this block-wise.
template <typename T>
void RowToColMajor(const T *Arm, size_t m, size_t n, T *Acm) {
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      Acm[j * m + i] = Arm[i * n + j];
}

// Populates a vector of function objects from a matlab struct
// containing the fields (f, a, b, c, d). The latter 4 are optional,
// while f is required. Each field (if present) is a vector of length n.
template <typename T>
int PopulateFunctionObj(const char fn_name[], const mxArray *f_mex,
                        unsigned int n, std::vector<FunctionObj<T>> *f_admm) {
  char alpha[] = "f\0a\0b\0c\0d\0";

  int f_idx = mxGetFieldNumber(f_mex, &alpha[0]);
  int a_idx = mxGetFieldNumber(f_mex, &alpha[2]);
  int b_idx = mxGetFieldNumber(f_mex, &alpha[4]);
  int c_idx = mxGetFieldNumber(f_mex, &alpha[6]);
  int d_idx = mxGetFieldNumber(f_mex, &alpha[8]);

  if (f_idx == -1) {
      mexErrMsgIdAndTxt("MATLAB:solver:missingParam",
          "Field %s.f is required.", fn_name);
      return 1;
  }

  void *f_data = 0, *a_data = 0, *b_data = 0, *c_data = 0, *d_data = 0;
  mxClassID f_id, a_id, b_id, c_id, d_id;

  // Find index and pointer to data of (f, a, b, c, d) in struct if present.
  mxArray *f_arr = mxGetFieldByNumber(f_mex, 0, f_idx);
  f_data = mxGetPr(f_arr);
  f_id = mxGetClassID(f_arr);
  if (!(mxGetM(f_arr) == n && mxGetN(f_arr) == 1) &&
      !(mxGetN(f_arr) == n && mxGetM(f_arr) == 1)) {
    mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
        "Dimension of %s.f and corresponding dimension of A must match.",
        fn_name);
    return 1;
  }
  if (a_idx != -1) {
    mxArray *a_arr = mxGetFieldByNumber(f_mex, 0, a_idx);
    a_data = mxGetPr(a_arr);
    a_id = mxGetClassID(a_arr);
    if (!(mxGetM(a_arr) == n && mxGetN(a_arr) == 1) &&
        !(mxGetN(a_arr) == n && mxGetM(a_arr) == 1)) {
      mexErrMsgIdAndTxt("MATLAB:solver:dimensionMismatch",
          "Dimension of %s.a and [f/g].f must match.", fn_name);
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
          "Dimension of %s.b and [f/g].f must match.", fn_name);
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
          "Dimension of %s.c and [f/g].f must match.", fn_name);
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
          "Dimension of %s.d and [f/g].f must match.", fn_name);
      return 1;
    }
  }

  // Populate f_admm.
  for (unsigned int i = 0; i < n; ++i) {
    T a = static_cast<T>(1);
    T b = static_cast<T>(0);
    T c = static_cast<T>(1);
    T d = static_cast<T>(0);
    Function f;

    // f may be of class double/float/int[32/64]/uint[32/64].
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
          "Function type %s.f must be double/float/int[32/64]/uint[32/64].",
          fn_name);
      return 1;
    }

    // (a, b, c, d) must be of class double or float.
    if (a_data != 0 && a_id == mxDOUBLE_CLASS) {
      a = static_cast<T>(reinterpret_cast<double*>(a_data)[i]);
    } else if (a_data != 0 && a_id ==  mxSINGLE_CLASS) {
      a = static_cast<T>(reinterpret_cast<float*>(a_data)[i]);
    } else if (a_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter %s.a must be double or single.", fn_name);
      return 1;
    }
    if (b_data != 0 && b_id == mxDOUBLE_CLASS) {
      b = static_cast<T>(reinterpret_cast<double*>(b_data)[i]);
    } else if (b_data != 0 && b_id == mxSINGLE_CLASS) {
      b = static_cast<T>(reinterpret_cast<float*>(b_data)[i]);
    } else if (b_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter %s.b must be double or single.", fn_name);
      return 1;
    }
    if (c_data != 0 && c_id == mxDOUBLE_CLASS) {
      c = static_cast<T>(reinterpret_cast<double*>(c_data)[i]);
    } else if (c_data != 0 && c_id == mxSINGLE_CLASS) {
      c = static_cast<T>(reinterpret_cast<float*>(c_data)[i]);
    } else if (c_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter %s.c must be double or single.", fn_name);
      return 1;
    }
    if (d_data != 0 && d_id == mxDOUBLE_CLASS) {
      d = static_cast<T>(reinterpret_cast<double*>(d_data)[i]);
    } else if (d_data != 0 && d_id == mxSINGLE_CLASS) {
      d = static_cast<T>(reinterpret_cast<float*>(d_data)[i]);
    } else if (d_data != 0) {
      mexErrMsgIdAndTxt("MATLAB:solver:inputNotNumeric",
          "Function parameter %s.d must be double or single.", fn_name);
      return 1;
    }

    f_admm->emplace_back(f, a, b, c, d);
  }
  return 0;
}

// Wrapper for graph solver. Populates admm_data structure and calls solver.
template <typename T>
void SolverWrap(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  size_t m = mxGetM(prhs[0]);
  size_t n = mxGetN(prhs[0]);

  // Convert column major (matlab) to row major (c++).
  T* A = new T[m * n];
  RowToColMajor(reinterpret_cast<T*>(mxGetPr(prhs[0])), n, m, A);

  AdmmData<T, T*> admm_data(A, m, n);
  admm_data.f.reserve(mxGetM(prhs[0]));
  admm_data.g.reserve(mxGetN(prhs[0]));
  admm_data.x = reinterpret_cast<T*>(mxGetPr(plhs[0]));
  if (nlhs == 2)
    admm_data.y = reinterpret_cast<T*>(mxGetPr(plhs[1]));

  int err = PopulateFunctionObj("f", prhs[1], m, &admm_data.f);
  if (err == 0)
    err = PopulateFunctionObj("g", prhs[2], n, &admm_data.g);

  if (err == 0)
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

