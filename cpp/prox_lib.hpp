#ifndef PROX_LIB_HPP_
#define PROX_LIB_HPP_

#include <cmath>
#include <limits>
#include <vector>

enum Function { kAbs,       // f(x) = |x|
                kHuber,     // f(x) = huber(x)
                kIdentity,  // f(x) = x
                kIndBox01,  // f(x) = I(0 <= x <= 1)
                kIndEq0,    // f(x) = I(x = 0)
                kIndGe0,    // f(x) = I(x >= 0)
                kIndLe0,    // f(x) = I(x <= 0)
                kNegLog,    // f(x) = -log(x)
                kLogistic,  // f(x) = log(1 + e^x)
                kMaxNeg0,   // f(x) = max(0, -x)
                kMaxPos0,   // f(x) = max(0, x)
                kSquare,    // f(x) = (1/2) x^2
                kZero };    // f(x) = 0


/**
 * Object associated with the generic function c * f(a * x - b) + d * x.
 */
template <typename T>
struct FunctionObj {
  Function f;
  T a, b, c, d;
  FunctionObj(Function f, T a, T b, T c, T d) : f(f), a(a), b(b), c(c), d(d) { }
  FunctionObj(Function f, T a, T b, T c) : f(f), a(a), b(b), c(c),
      d(static_cast<T>(0)) { }
  FunctionObj(Function f, T a, T b) : f(f), a(a), b(b),
      c(static_cast<T>(1)), d(static_cast<T>(0)) { }
  FunctionObj(Function f, T a) : f(f), a(a),
      b(static_cast<T>(0)), c(static_cast<T>(1)), d(static_cast<T>(0)) { }
  explicit FunctionObj(Function f) : f(f), a(static_cast<T>(1)),
      b(static_cast<T>(0)), c(static_cast<T>(1)), d(static_cast<T>(0)) { }
};


/* Useful Local Functions */
namespace {
/**
 * Evalution of max(0, x).
 */
template <typename T>
inline T MaxPos(T x) {
  return x >= static_cast<T>(0) ? x : static_cast<T>(0);
}

/**
 * Evalution of max(0, -x).
 */
template <typename T>
inline T MaxNeg(T x) {
  return x <= static_cast<T>(0) ? -x : static_cast<T>(0);
}
}  // namespace


/* Proximal Operator Definitions */

template <typename T>
T ProxAbs(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z  = MaxPos(x_ - static_cast<T>(1) / rho_) - MaxNeg(x_ + static_cast<T>(1));
  return (z + b) / a;
}

template <typename T>
T ProxHuber(T x, T a, T b, T c, T d, T rho) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T ProxIdentity(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z  = x_ - static_cast<T>(1) / rho_;
  return (z + b) / a;
}

template <typename T>
T ProxIndBox01(T x, T a, T b, T c, T d, T rho) {
  x = x - d / rho;
  x = a * x <= b ? static_cast<T>(0) : a * x - b;
  x = a * x >= b + static_cast<T>(1) ? static_cast<T>(1) : a * x - b;
  return x;
}

template <typename T>
T ProxIndEq0(T x, T a, T b, T c, T d, T rho) {
  return b / a;
}

template <typename T>
T ProxIndGe0(T x, T a, T b, T c, T d, T rho) {
  return a * (x - d / rho) <= b ? static_cast<T>(0) : a * (x - d / rho) - b;
}

template <typename T>
T ProxIndLe0(T x, T a, T b, T c, T d, T rho) {
  return a * (x - d / rho) >= b ? static_cast<T>(0) : a * (x - d / rho) - b;
}

template <typename T>
T ProxNegLog(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = (x_ + sqrt(x_ * x_ + 4 / rho_)) / 2;
  return (z + b) / a;
}

template <typename T>
T ProxLogistic(T x, T a, T b, T c, T d, T rho) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T ProxMaxNeg0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = x_ >= static_cast<T>(0) ? x_ : static_cast<T>(0);
  z = x_ <= -static_cast<T>(1) / rho_
      ? x_ + static_cast<T>(1) / rho_ : z;
  return (z + b) / a;
}

template <typename T>
T ProxMaxPos0(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z = x_ <= static_cast<T>(0) ? x_ : static_cast<T>(0);
  z = x_ >= static_cast<T>(1) / rho_
      ? x_ - static_cast<T>(1) / rho_ : z;
  return (z + b) / a;
}

template <typename T>
T ProxSquare(T x, T a, T b, T c, T d, T rho) {
  T x_ = a * (x - d / rho) - b;
  T rho_ = rho / (c * a * a);
  T z  = rho_ * x_ / (static_cast<T>(1) + rho_);
  return (z + b) / a;
}

template <typename T>
T ProxZero(T x, T a, T b, T c, T d, T rho) {
  return x - d / rho;
}


/* Function Definitions */

template <typename T>
T FuncAbs(T x, T a, T b, T c, T d) {
  return c * fabs(a * x + b);
}

template <typename T>
T FuncHuber(T x, T a, T b, T c, T d) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T FuncIdentity(T x, T a, T b, T c, T d) {
  return c * (a * x + b);
}

template <typename T>
T FuncIndBox01(T x, T a, T b, T c, T d) {
  return 0;
}

template <typename T>
T FuncIndEq0(T x, T a, T b, T c, T d) {
  return 0;
}

template <typename T>
T FuncIndGe0(T x, T a, T b, T c, T d) {
  return 0;
}

template <typename T>
T FuncIndLe0(T x, T a, T b, T c, T d) {
  return 0;
}

template <typename T>
T FuncNegLog(T x, T a, T b, T c, T d) {
  return -c * log(a * x - b);
}

template <typename T>
T FuncLogistic(T x, T a, T b, T c, T d) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T FuncMaxNeg0(T x, T a, T b, T c, T d) {
  return c * MaxNeg(a * x - b);
}

template <typename T>
T FuncMaxPos0(T x, T a, T b, T c, T d) {
  return c * MaxPos(a * x - b);
}

template <typename T>
T FuncSquare(T x, T a, T b, T c, T d) {
  T sq = (a * x - b);
  return c * sq * sq / 2;
}

template <typename T>
T FuncZero(T x, T a, T b, T c, T d) {
  return 0;
}


/**
 * Evaluates proximal operator x_out <- f_obj.prox(x_in) element-wise.
 *
 * @param f_obj Vector of function objects.
 * @param rho Penalty parameter.
 * @param x_in Array to which proximal operator will be applied.
 * @param x_out Array to which result will be written.
 */
template <typename T>
void ProxEval(const std::vector<FunctionObj<T>> f_obj, T rho, const T* x_in,
              T* x_out) {
  #pragma omp parallel for
  for (unsigned int i = 0; i < f_obj.size(); ++i) {
    const T x = x_in[i];
    const T a = f_obj[i].a;
    const T b = f_obj[i].b;
    const T c = f_obj[i].c;
    const T d = f_obj[i].d;
    switch (f_obj[i].f) {
      case kAbs:
        x_out[i] = ProxAbs(x, a, b, c, d, rho); break;
      case kHuber:
        x_out[i] = ProxHuber(x, a, b, c, d, rho); break;
      case kIdentity:
        x_out[i] = ProxIdentity(x, a, b, c, d, rho); break;
      case kIndBox01:
        x_out[i] = ProxIndBox01(x, a, b, c, d, rho); break;
      case kIndEq0:
        x_out[i] = ProxIndEq0(x, a, b, c, d, rho); break;
      case kIndGe0:
        x_out[i] = ProxIndGe0(x, a, b, c, d, rho); break;
      case kIndLe0:
        x_out[i] = ProxIndLe0(x, a, b, c, d, rho); break;
      case kNegLog:
        x_out[i] = ProxNegLog(x, a, b, c, d, rho); break;
      case kLogistic:
        x_out[i] = ProxLogistic(x, a, b, c, d, rho); break;
      case kMaxNeg0:
        x_out[i] = ProxMaxNeg0(x, a, b, c, d, rho); break;
      case kMaxPos0:
        x_out[i] = ProxMaxPos0(x, a, b, c, d, rho); break;
      case kSquare:
        x_out[i] = ProxSquare(x, a, b, c, d, rho); break;
      case kZero:
        x_out[i] = ProxZero(x, a, b, c, d, rho); break;
    }
  }
}

/**
 * Evaluates function x_out <- f_obj.func(x_in) element-wise.
 *
 * @param f_obj Vector of function objects.
 * @param x_in Array to which function will be applied.
 * @param x_out Array to which result will be written.
 */
template <typename T>
T FuncEval(const std::vector<FunctionObj<T>> f_obj, T rho, const T* x_in) {
  T sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (unsigned int i = 0; i < f_obj.size(); ++i) {
    const T x = x_in[i];
    const T a = f_obj[i].a;
    const T b = f_obj[i].b;
    const T c = f_obj[i].c;
    const T d = f_obj[i].d;
    switch (f_obj[i].f) {
      case kAbs:
        sum += FuncAbs(x, a, b, c, d); break;
      case kHuber:
        sum += FuncHuber(x, a, b, c, d); break;
      case kIdentity:
        sum += FuncIdentity(x, a, b, c, d); break;
      case kIndBox01:
        sum += FuncIndBox01(x, a, b, c, d); break;
      case kIndEq0:
        sum += FuncIndEq0(x, a, b, c, d); break;
      case kIndGe0:
        sum += FuncIndGe0(x, a, b, c, d); break;
      case kIndLe0:
        sum += FuncIndLe0(x, a, b, c, d); break;
      case kNegLog:
        sum += FuncNegLog(x, a, b, c, d); break;
      case kLogistic:
        sum += FuncLogistic(x, a, b, c, d); break;
      case kMaxNeg0:
        sum += FuncMaxNeg0(x, a, b, c, d); break;
      case kMaxPos0:
        sum += FuncMaxPos0(x, a, b, c, d); break;
      case kSquare:
        sum += FuncSquare(x, a, b, c, d); break;
      case kZero:
        sum += FuncZero(x, a, b, c, d); break;
    }
  }
  return sum;
}

#endif /* PROX_LIB_HPP_ */

