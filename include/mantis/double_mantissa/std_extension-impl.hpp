/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_STD_EXTENSION_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_STD_EXTENSION_IMPL_H_

#include "mantis/double_mantissa.hpp"

namespace std {

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::epsilon() {
  return mantis::double_mantissa::Epsilon<float>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::infinity() {
  return mantis::double_mantissa::Infinity<float>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::quiet_NaN() {
  return mantis::double_mantissa::QuietNan<float>();
}

constexpr mantis::DoubleMantissa<float>
numeric_limits<mantis::DoubleMantissa<float>>::signaling_NaN() {
  return mantis::double_mantissa::SignalingNan<float>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::epsilon() {
  return mantis::double_mantissa::Epsilon<double>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::infinity() {
  return mantis::double_mantissa::Infinity<double>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::quiet_NaN() {
  return mantis::double_mantissa::QuietNan<double>();
}

constexpr mantis::DoubleMantissa<double>
numeric_limits<mantis::DoubleMantissa<double>>::signaling_NaN() {
  return mantis::double_mantissa::SignalingNan<double>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::epsilon() {
  return mantis::double_mantissa::Epsilon<long double>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::infinity() {
  return mantis::double_mantissa::Infinity<long double>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::quiet_NaN() {
  return mantis::double_mantissa::QuietNan<long double>();
}

constexpr mantis::DoubleMantissa<long double>
numeric_limits<mantis::DoubleMantissa<long double>>::signaling_NaN() {
  return mantis::double_mantissa::SignalingNan<long double>();
}

template <typename Real>
mantis::DoubleMantissa<Real> abs(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Abs(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> acos(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcCos(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> acosh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcHyperbolicCos(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> asin(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcSin(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> asinh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcHyperbolicSin(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> atan(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcTan(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> atan2(const mantis::DoubleMantissa<Real>& y,
                                   const mantis::DoubleMantissa<Real>& x) {
  return mantis::ArcTan2(y, x);
}

template <typename Real>
mantis::DoubleMantissa<Real> atanh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::ArcHyperbolicTan(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> cos(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Cos(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> cosh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::HyperbolicCos(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> exp(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Exp(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> floor(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Floor(value);
}

template <typename Real>
bool isfinite(const mantis::DoubleMantissa<Real>& value) {
  return isfinite(value.Upper()) && isfinite(value.Lower());
}

template <typename Real>
bool isinf(const mantis::DoubleMantissa<Real>& value) {
  return isinf(value.Upper()) || isinf(value.Lower());
}

template <typename Real>
bool isnan(const mantis::DoubleMantissa<Real>& value) {
  return isnan(value.Upper()) || isnan(value.Lower());
}

template <typename Real>
mantis::DoubleMantissa<Real> ldexp(const mantis::DoubleMantissa<Real>& value,
                                   int exp) {
  return mantis::LoadExponent(value, exp);
}

template <typename Real>
mantis::DoubleMantissa<Real> log(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Log(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> log10(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Log10(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> pow(const mantis::DoubleMantissa<Real>& value,
                                 int exponent) {
  return mantis::IntegerPower(value, exponent);
}

template <typename Real>
mantis::DoubleMantissa<Real> pow(const mantis::DoubleMantissa<Real>& value,
                                 const mantis::DoubleMantissa<Real>& exponent) {
  return mantis::Power(value, exponent);
}

template <typename Real>
mantis::DoubleMantissa<Real> round(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Round(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> sin(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Sin(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> sinh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::HyperbolicSin(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> sqrt(const mantis::DoubleMantissa<Real>& value) {
  return mantis::SquareRoot(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> tan(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Tan(value);
}

template <typename Real>
mantis::DoubleMantissa<Real> tanh(const mantis::DoubleMantissa<Real>& value) {
  return mantis::HyperbolicTan(value);
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_STD_EXTENSION_IMPL_H_
