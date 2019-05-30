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
constexpr Real MultiplyAdd(const Real& x, const Real& y,
                           const Real& z) MANTIS_NOEXCEPT {
  return x * y + z;
}

inline constexpr float MultiplyAdd(const float& x, const float& y,
                                   const float& z) MANTIS_NOEXCEPT {
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

inline constexpr double MultiplyAdd(const double& x, const double& y,
                                    const double& z) MANTIS_NOEXCEPT {
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

inline constexpr long double MultiplyAdd(const long double& x,
                                         const long double& y,
                                         const long double& z) MANTIS_NOEXCEPT {
// The QD package of Hida et al. also allows for usage of the "_Asm_fma"
// macro on Intel and HP compilers for IA 64.
//
// There appears to be no equivalent of the XLC __fmadd function for
// long double.
#ifdef FP_FAST_FMAL
  return std::fmal(x, y, z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmal(x, y, z);
#else
  return std::fmal(x, y, z);
#endif
}

template <typename Real>
constexpr Real MultiplySubtract(const Real& x, const Real& y,
                                const Real& z) MANTIS_NOEXCEPT {
  return x * y - z;
}

inline constexpr float MultiplySubtract(const float& x, const float& y,
                                        const float& z) MANTIS_NOEXCEPT {
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

inline constexpr double MultiplySubtract(const double& x, const double& y,
                                         const double& z) MANTIS_NOEXCEPT {
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

inline constexpr long double MultiplySubtract(
    const long double& x, const long double& y,
    const long double& z) MANTIS_NOEXCEPT {
// The QD package of Hida et al. also allows for usage of the "_Asm_fms"
// macro on Intel and HP compilers for IA 64.
//
// There appears to be no equivalent of the XLC __fmsub function for
// long double.
#ifdef FP_FAST_FMAL
  return std::fmal(x, y, -z);
#elif defined(MANTIS_GCC_FUSED_MULTIPLY_ADD)
  return __builtin_fmal(x, y, -z);
#else
  return std::fmal(x, y, -z);
#endif
}

}  // namespace mantis

#endif  // ifndef MANTIS_UTIL_IMPL_H_
