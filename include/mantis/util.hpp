/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_UTIL_H_
#define MANTIS_UTIL_H_

#include <type_traits>

#include "mantis/enable_if.hpp"
#include "mantis/fpu_fix.hpp"

namespace mantis {

// Performs a -- preferably fused -- multiply and add of the form: x * y + z.
template <typename Real>
constexpr Real MultiplyAdd(const Real& x, const Real& y,
                           const Real& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Add to fp32.
float MultiplyAdd(const float& x, const float& y,
                  const float& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Add to fp64.
double MultiplyAdd(const double& x, const double& y,
                   const double& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Add to long double.
long double MultiplyAdd(const long double& x, const long double& y,
                        const long double& z) MANTIS_NOEXCEPT;

// Performs a -- preferably fused -- multiply subtract: x * y - z.
template <typename Real>
constexpr Real MultiplySubtract(const Real& x, const Real& y,
                                const Real& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Subtract to fp32.
float MultiplySubtract(const float& x, const float& y,
                       const float& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Subtract to fp64.
double MultiplySubtract(const double& x, const double& y,
                        const double& z) MANTIS_NOEXCEPT;

// A specialization of the Multiply Subtract to long double.
long double MultiplySubtract(const long double& x, const long double& y,
                             const long double& z) MANTIS_NOEXCEPT;

// An implementation of std::ceil which is constexpr.
template <typename Real>
constexpr long long int constexpr_ceil(const Real& value);

}  // namespace mantis

#include "mantis/util-impl.hpp"

#endif  // ifndef MANTIS_UTIL_H_
