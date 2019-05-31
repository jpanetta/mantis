/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_STD_NUMERIC_LIMITS_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_STD_NUMERIC_LIMITS_IMPL_H_

#include "mantis/double_mantissa.hpp"

namespace std {

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::lowest() {
  return numeric_limits<float>::lowest();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::min() {
  return numeric_limits<float>::min();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::max() {
  return numeric_limits<float>::max();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::epsilon() {
  return mantis::Epsilon<mantis::DoubleMantissa<float>>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::infinity() {
  return mantis::Infinity<mantis::DoubleMantissa<float>>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::quiet_NaN() {
  return mantis::QuietNan<mantis::DoubleMantissa<float>>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::signaling_NaN() {
  return mantis::SignalingNan<mantis::DoubleMantissa<float>>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::lowest() {
  return numeric_limits<double>::lowest();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::min() {
  return numeric_limits<double>::min();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::max() {
  return numeric_limits<double>::max();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::epsilon() {
  return mantis::Epsilon<mantis::DoubleMantissa<double>>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::infinity() {
  return mantis::Infinity<mantis::DoubleMantissa<double>>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::quiet_NaN() {
  return mantis::QuietNan<mantis::DoubleMantissa<double>>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::signaling_NaN() {
  return mantis::SignalingNan<mantis::DoubleMantissa<double>>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::lowest() {
  return numeric_limits<long double>::lowest();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::min() {
  return numeric_limits<long double>::min();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::max() {
  return numeric_limits<long double>::max();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::epsilon() {
  return mantis::Epsilon<mantis::DoubleMantissa<long double>>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::infinity() {
  return mantis::Infinity<mantis::DoubleMantissa<long double>>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::quiet_NaN() {
  return mantis::QuietNan<mantis::DoubleMantissa<long double>>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::signaling_NaN() {
  return mantis::SignalingNan<mantis::DoubleMantissa<long double>>();
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_STD_NUMERIC_LIMITS_IMPL_H_
