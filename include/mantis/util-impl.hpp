/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_UTIL_IMPL_H_
#define MANTIS_UTIL_IMPL_H_
#include <cmath>
#include <limits>

#include "mantis/util.hpp"

namespace mantis {

template <typename Real>
Real MultiplyAdd(const Real& x, const Real& y, const Real& z) {
  return x * y + z;
}

inline float MultiplyAdd(const float& x, const float& y, const float& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fma"
// macro on Intel and HP compilers for IA 64.
#ifdef FP_FAST_FMAF
  return std::fmaf(x, y, z)
#elif defined(MANTIS_XLC_FUSED_MULTIPLY_ADD)
  return __fmadds(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmaf(x, y, z);
#else
  return std::fmaf(x, y, z);
#endif
}

inline double MultiplyAdd(const double& x, const double& y, const double& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fma"
// macro on Intel and HP compilers for IA 64.
#ifdef FP_FAST_FMA
  return std::fma(x, y, z);
#elif defined(MANTIS_XLC_FUSED_MULTIPLY_ADD)
  return __fmadd(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fma(x, y, z);
#else
  return std::fma(x, y, z);
#endif
}

inline long double MultiplyAdd(const long double& x, const long double& y,
                               const long double& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fma"
// macro on Intel and HP compilers for IA 64.
//
// There appears to be no equivalent of the XLC __fmadd function for
// long double.
#ifdef FP_FAST_FMA
  return std::fma(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmal(x, y, z);
#else
  return std::fma(x, y, z);
#endif
}

template <typename Real>
Real MultiplySubtract(const Real& x, const Real& y, const Real& z) {
  return x * y - z;
}

inline float MultiplySubtract(const float& x, const float& y, const float& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fms"
// macro on Intel and HP compilers for IA 64.
#ifdef FP_FAST_FMAF
  return std::fmaf(x, y, -z);
#elif defined(MANTIS_XLC_FUSED_MULTIPLY_ADD)
  return __fmsubs(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmaf(x, y, -z);
#else
  return std::fmaf(x, y, -z);
#endif
}

inline double MultiplySubtract(const double& x, const double& y,
                               const double& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fms"
// macro on Intel and HP compilers for IA 64.
#ifdef FP_FAST_FMA
  return std::fma(x, y, -z);
#elif defined(MANTIS_XLC_FUSED_MULTIPLY_ADD)
  return __fmsub(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fma(x, y, -z);
#else
  return std::fma(x, y, -z);
#endif
}

inline long double MultiplySubtract(const long double& x, const long double& y,
                                    const long double& z) {
// The QD package of Hida et al. also allows for usage of the "_Asm_fms"
// macro on Intel and HP compilers for IA 64.
//
// There appears to be no equivalent of the XLC __fmsub function for
// long double.
#ifdef FP_FAST_FMA
  return std::fma(x, y, -z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmal(x, y, -z);
#else
  return std::fma(x, y, -z);
#endif
}

template <typename Real>
Real QuickTwoSum(const Real& larger, const Real& smaller, Real* error) {
  const Real result = larger + smaller;
  *error = smaller - (result - larger);
  return result;
}

template <typename Real>
Real TwoSum(const Real& larger, const Real& smaller, Real* error) {
  const Real result = larger + smaller;
  const Real smaller_approx = result - larger;
  *error = (larger - (result - smaller_approx)) + (smaller - smaller_approx);
  return result;
}

template <typename Real>
Real QuickTwoDiff(const Real& larger, const Real& smaller, Real* error) {
  const Real result = larger - smaller;
  *error = (larger - result) - smaller;
  return result;
}

template <typename Real>
Real TwoDiff(const Real& larger, const Real& smaller, Real* error) {
  const Real result = larger - smaller;
  const Real smaller_approx = larger - result;
  *error = (larger - (result + smaller_approx)) - (smaller - smaller_approx);
  return result;
}

template <typename Real>
Real TwoProdFMA(const Real& x, const Real& y, Real* error) {
  const Real product = x * y;
  *error = MultiplySubtract(x, y, product);
  return product;
}

template <typename Real>
void Split(const Real& value, Real* high, Real* low) {
  static const int num_digits = std::numeric_limits<Real>::digits;
  static const int safe_max = std::numeric_limits<Real>::max();

  static const Real kSplitter = (1u << num_digits) + 1;
  static const Real kSplitScale = 1u << (num_digits + 1);
  static const Real kSplitThreshold = safe_max / kSplitScale;

  if (value > kSplitThreshold || value < -kSplitThreshold) {
    static const Real kSplitInvScale = Real{1} / kSplitScale;
    const Real scaled_value = kSplitInvScale * value;

    const Real temp = kSplitter * value;
    *high = temp - (temp - value);
    *low = value - high;

    *high *= kSplitScale;
    *low *= kSplitScale;
  } else {
    const Real temp = kSplitter * value;
    *high = temp - (temp - value);
    *low = value - high;
  }
}

template <typename Real>
Real TwoProd(const Real& x, const Real& y, Real* error) {
  const Real product = x * y;

  Real x_high, x_low;
  Split(x, &x_high, &x_low);

  Real y_high, y_low;
  Split(y, &y_high, &y_low);

  *error = ((x_high * y_high - product) + x_high * y_low + x_low * y_high) +
           x_low * y_low;

  return product;
}

inline float TwoProd(const float& larger, const float& smaller, float* error) {
  return TwoProdFMA(larger, smaller, error);
}

inline double TwoProd(const double& larger, const double& smaller,
                      double* error) {
  return TwoProdFMA(larger, smaller, error);
}

inline long double TwoProd(const long double& larger,
                           const long double& smaller, long double* error) {
  return TwoProdFMA(larger, smaller, error);
}

template <typename Real>
Real TwoSquare(const Real& x, Real* error) {
  const Real product = x * x;
  Real high, low;
  Split(x, &high, &low);
  *error = ((high * high - product) + Real{2} * high * low) + low * low;
  return product;
}

inline float TwoSquare(const float& x, float* error) {
  const float product = x * x;
  *error = MultiplySubtract(x, x, product);
  return product;
}

inline double TwoSquare(const double& x, double* error) {
  const double product = x * x;
  *error = MultiplySubtract(x, x, product);
  return product;
}

inline long double TwoSquare(const long double& x, long double* error) {
  const long double product = x * x;
  *error = MultiplySubtract(x, x, product);
  return product;
}

// DO_NOT_SUBMIT
#define X86
#define FPU_GETCW(x) asm volatile("fnstcw %0" : "=m"(x));
#define FPU_SETCW(x) asm volatile("fldcw %0" : : "m"(x));
#define FPU_EXTENDED 0x0300
#define FPU_DOUBLE 0x0200

inline FPUFix::FPUFix() {
#ifdef X86
  volatile unsigned short control_word, new_control_word;
  FPU_GETCW(control_word);
  new_control_word = (control_word & ~FPU_EXTENDED) | FPU_DOUBLE;
  FPU_SETCW(new_control_word);
  if (control_word_) {
    control_word_ = control_word;
  }
#endif  // ifdef X86
}

inline FPUFix::~FPUFix() {
#ifdef X86
  if (control_word_) {
    int control_word;
    control_word = control_word_;
    FPU_SETCW(control_word);
  }
#endif  // ifdef X86
}

}  // namespace mantis

#endif  // ifndef MANTIS_UTIL_IMPL_H_
