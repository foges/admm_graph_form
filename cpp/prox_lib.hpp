#ifndef PROX_LIB_
#define PROX_LIB_

#include <cmath>
#include <limits>
#include <vector>

enum Function { kAbs,      // f(x) = |x|
                kHuber,    // f(x) = huber(x)
                kIdentity, // f(x) = x
                kIndBox01, // f(x) = I(0 <= x <= 1)
                kIndEq0,   // f(x) = I(x = 0)
                kIndGe0,   // f(x) = I(x >= 0)
                kIndLe0,   // f(x) = I(x <= 0)
                kNegLog,   // f(x) = -log(x)
                kLogistic, // f(x) = log(1 + e^x)
                kMaxNeg0,  // f(x) = max(0, -x) 
                kMaxPos0,  // f(x) = max(0, x)
                kSquare,   // f(x) = (1/2) x^2
                kZero};    // f(x) = 0

template <typename T>
struct FunctionObj {
  Function f;
  T a, b, c;
  FunctionObj(Function f, T a, T b, T c) : f(f), a(a), b(b), c(c) { }
};


/* Useful Local Functions */
namespace {
template <typename T>
inline T MaxPos(T x) {
  return x >= static_cast<T>(0) ? x : static_cast<T>(0);
}

template <typename T>
inline T MaxNeg(T x) {
  return x <= static_cast<T>(0) ? -x : static_cast<T>(0);
}
}


/* Proximal Operator Definitions */
template <typename T>
T ProxAbs(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z  = MaxPos(x_ - static_cast<T>(1) / rho_) - MaxNeg(x_ + static_cast<T>(1));
  return (z + b) / a;
}

template <typename T>
T ProxHuber(T x, T a, T b, T c, T rho) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T ProxIdentity(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z  = x_ - static_cast<T>(1) / rho_;
  return (z + b) / a;
}

template <typename T>
T ProxIndBox01(T x, T a, T b, T c, T rho) {
  x = a * x <= b ? static_cast<T>(0) : a * x - b;
  x = a * x >= b + static_cast<T>(1) ? static_cast<T>(1) : a * x - b;
  return x;
}

template <typename T>
T ProxIndEq0(T x, T a, T b, T c, T rho) {
  return b / a;
}

template <typename T>
T ProxIndGe0(T x, T a, T b, T c, T rho) {
  return a * x <= b ? static_cast<T>(0) : a * x - b;
}

template <typename T>
T ProxIndLe0(T x, T a, T b, T c, T rho) {
  return a * x >= b ? static_cast<T>(0) : a * x - b;
}

template <typename T>
T ProxNegLog(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z = (x_ + sqrt(x_ * x_ + 4 / rho_)) / 2;
  return (z + b) / a;
}

template <typename T>
T ProxLogistic(T x, T a, T b, T c, T rho) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T ProxMaxNeg0(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z = x_ >= static_cast<T>(0) ? x_ : static_cast<T>(0);
  z = x_ <= -static_cast<T>(1) / rho_
      ? x_ + static_cast<T>(1) / rho_ : z;
  return (z + b) / a;
}

template <typename T>
T ProxMaxPos0(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z = x_ <= static_cast<T>(0) ? x_ : static_cast<T>(0);
  z = x_ >= static_cast<T>(1) / rho_
      ? x_ - static_cast<T>(1) / rho_ : z;
  return (z + b) / a;
}

template <typename T>
T ProxSquare(T x, T a, T b, T c, T rho) {
  T rho_ = rho / (c * a * a);
  T x_ = a * x - b;
  T z  = rho_ * x_ / (static_cast<T>(1) + rho_);
  return (z + b) / a;
}

template <typename T>
T ProxZero(T x, T a, T b, T c, T rho) {
  return x;
}


/* Function Definitions */
template <typename T>
T FuncAbs(T x, T a, T b, T c) {
  return c * fabs(a * x + b);
}

template <typename T>
T FuncHuber(T x, T a, T b, T c) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T FuncIdentity(T x, T a, T b, T c) {
  return c * (a * x + b);
}

template <typename T>
T FuncIndBox01(T x, T a, T b, T c) {
  return 0;
}

template <typename T>
T FuncIndEq0(T x, T a, T b, T c) {
  return 0;
}

template <typename T>
T FuncIndGe0(T x, T a, T b, T c) {
  return 0;
}

template <typename T>
T FuncIndLe0(T x, T a, T b, T c) {
  return 0;
}

template <typename T>
T FuncNegLog(T x, T a, T b, T c) {
  return -c * log(a * x - b);
}

template <typename T>
T FuncLogistic(T x, T a, T b, T c) {
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T FuncMaxNeg0(T x, T a, T b, T c) {
  return c * MaxNeg(a * x - b);
}

template <typename T>
T FuncMaxPos0(T x, T a, T b, T c) {
  return c * MaxPos(a * x - b);
}

template <typename T>
T FuncSquare(T x, T a, T b, T c) {
  T sq = (a * x - b);
  return c * sq * sq / 2;
}

template <typename T>
T FuncZero(T x, T a, T b, T c) {
  return 0;
}


/* Evaluates Proximal Functions */
template <typename T>
void ProxEval(const std::vector<FunctionObj<T>> f_obj, T rho, const T* x_in,
              T* x_out) {
  #pragma omp parallel for
  for (unsigned int i = 0; i < f_obj.size(); ++i) {
    const T x = x_in[i];
    const T a = f_obj[i].a;
    const T b = f_obj[i].b;
    const T c = f_obj[i].c;
    switch (f_obj[i].f) {
      case kAbs:
        x_out[i] = ProxAbs(x, a, b, c, rho); break;
      case kHuber:
        x_out[i] = ProxHuber(x, a, b, c, rho); break;
      case kIdentity:
        x_out[i] = ProxIdentity(x, a, b, c, rho); break;
      case kIndBox01:
        x_out[i] = ProxIndBox01(x, a, b, c, rho); break;
      case kIndEq0:
        x_out[i] = ProxIndEq0(x, a, b, c, rho); break;
      case kIndGe0:
        x_out[i] = ProxIndGe0(x, a, b, c, rho); break;
      case kIndLe0:
        x_out[i] = ProxIndLe0(x, a, b, c, rho); break;
      case kNegLog:
        x_out[i] = ProxNegLog(x, a, b, c, rho); break;
      case kLogistic:
        x_out[i] = ProxLogistic(x, a, b, c, rho); break;
      case kMaxNeg0:
        x_out[i] = ProxMaxNeg0(x, a, b, c, rho); break;
      case kMaxPos0:
        x_out[i] = ProxMaxPos0(x, a, b, c, rho); break;
      case kSquare:
        x_out[i] = ProxSquare(x, a, b, c, rho); break;
      case kZero:
        x_out[i] = ProxZero(x, a, b, c, rho); break;
    }
  }
}

#endif /* PROX_LIB_ */

