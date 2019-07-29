/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_CLASS_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_CLASS_IMPL_H_

#include <random>

#include "mantis/util.hpp"

#include "mantis/double_mantissa.hpp"

namespace mantis {

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa() MANTIS_NOEXCEPT
    : values_{Real{0}, Real{0}} {}

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa(const Real& upper)
    MANTIS_NOEXCEPT : values_{upper, Real{0}} {}

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa(const Real& upper,
                                               const Real& lower)
    MANTIS_NOEXCEPT : values_{upper, lower} {}

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT
    : values_{value.Upper(), value.Lower()} {}

template <typename Real>
DoubleMantissa<Real>::DoubleMantissa(const BinaryNotation& rep) {
  FromBinary(rep);
}

template <typename Real>
DoubleMantissa<Real>::DoubleMantissa(const DecimalNotation& rep) {
  FromDecimal(rep);
}

template <typename Real>
DoubleMantissa<Real>::DoubleMantissa(const std::string& rep) {
  DecimalNotation scientific;
  scientific.FromString(rep);
  FromDecimal(scientific);
}

template <typename Real>
constexpr Real& DoubleMantissa<Real>::Upper() MANTIS_NOEXCEPT {
  return values_[0];
}

template <typename Real>
constexpr const Real& DoubleMantissa<Real>::Upper() const MANTIS_NOEXCEPT {
  return values_[0];
}

template <typename Real>
constexpr Real& DoubleMantissa<Real>::Lower() MANTIS_NOEXCEPT {
  return values_[1];
}

template <typename Real>
constexpr const Real& DoubleMantissa<Real>::Lower() const MANTIS_NOEXCEPT {
  return values_[1];
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::Reduce() MANTIS_NOEXCEPT {
  *this = TwoSum(*this);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(int value)
    MANTIS_NOEXCEPT {
  Upper() = value;
  Lower() = 0;
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(long int value)
    MANTIS_NOEXCEPT {
  Upper() = value;
  Lower() = 0;
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(
    long long int value) MANTIS_NOEXCEPT {
  Upper() = value;
  Lower() = 0;
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(
    const Real& value) MANTIS_NOEXCEPT {
  Upper() = value;
  Lower() = 0;
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT {
  Upper() = value.Upper();
  Lower() = value.Lower();
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(int value)
    MANTIS_NOEXCEPT {
  return *this += Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(long int value)
    MANTIS_NOEXCEPT {
  return *this += Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(
    long long int value) MANTIS_NOEXCEPT {
  return *this += Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(
    const Real& value) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> sum = TwoSum(Upper(), value);
  sum.Lower() += Lower();
  *this = QuickTwoSum(sum);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT {
#ifdef MANTIS_IEEE_SUM
  // This algorithm is attributed by Hida et al. to Briggs and Kahan.
  DoubleMantissa<Real> upper_sum = TwoSum(Upper(), value.Upper());
  const DoubleMantissa<Real> lower_sum = TwoSum(Lower(), value.Lower());
  upper_sum.Lower() += lower_sum.Upper();

  upper_sum = QuickTwoSum(upper_sum);
  upper_sum.Lower() += lower_sum.Lower();

  *this = QuickTwoSum(upper_sum);
#else
  // In QD, this is referred to as a 'sloppy' add, as it only obey's a
  // Cray-style error bound.
  DoubleMantissa<Real> upper_sum = TwoSum(Upper(), value.Upper());
  upper_sum.Lower() += Lower();
  upper_sum.Lower() += value.Lower();

  *this = QuickTwoSum(upper_sum);
#endif  // ifdef MANTIS_IEEE_SUM
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(int value)
    MANTIS_NOEXCEPT {
  return *this -= Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(long int value)
    MANTIS_NOEXCEPT {
  return *this -= Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(
    long long int value) MANTIS_NOEXCEPT {
  return *this -= Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(
    const Real& value) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> upper_diff = TwoDiff(Upper(), value);
  upper_diff.Lower() += Lower();

  *this = QuickTwoSum(upper_diff);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT {
#ifdef MANTIS_IEEE_SUM
  DoubleMantissa<Real> upper_diff = TwoDiff(Upper(), value.Upper());
  const DoubleMantissa<Real> lower_diff = TwoDiff(Lower(), value.Lower());
  upper_diff.Lower() += lower_diff.Upper();

  upper_diff = QuickTwoSum(upper_diff);
  upper_diff.Lower() += lower_diff.Lower();

  *this = QuickTwoSum(upper_diff);
#else
  DoubleMantissa<Real> upper_diff = TwoDiff(Upper(), value.Upper());
  upper_diff.Lower() += Lower();
  upper_diff.Lower() -= value.Lower();

  *this = QuickTwoSum(upper_diff);
#endif  // ifdef MANTIS_IEEE_SUM
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(int value)
    MANTIS_NOEXCEPT {
  return *this *= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(long int value)
    MANTIS_NOEXCEPT {
  return *this *= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(long long int value)
    MANTIS_NOEXCEPT {
  return *this *= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(const Real& value)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> product = TwoProd(Upper(), value);
  product.Lower() += Lower() * value;
  *this = QuickTwoSum(product);
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> product = TwoProd(Upper(), value.Upper());
  product.Lower() += value.Lower() * Upper();
  product.Lower() += value.Upper() * Lower();
  *this = QuickTwoSum(product);
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(int value) {
  return *this /= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(long int value) {
  return *this /= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(long long int value) {
  return *this /= Real(value);
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(const Real& value) {
  // See the QD library by Hida et al. for a similar implementation.
  const Real approx_quotient = Upper() / value;

  // Compute the remainder, remander := *this - approx_quotient * value.
  DoubleMantissa<Real> product = TwoProd(approx_quotient, value);
  DoubleMantissa<Real> remainder = TwoDiff(Upper(), product.Upper());
  remainder.Lower() += Lower();
  remainder.Lower() -= product.Lower();

  // Perform iterative refinement.
  const Real update = (remainder.Upper() + remainder.Lower()) / value;
  *this = QuickTwoSum(approx_quotient, update);

  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator int() const MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return int(floored_value.Upper()) + int(floored_value.Lower());
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator long int() const MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return static_cast<long int>(floored_value.Upper()) +
         static_cast<long int>(floored_value.Lower());
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator long long int() const MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return static_cast<long long int>(floored_value.Upper()) +
         static_cast<long long int>(floored_value.Lower());
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator float() const MANTIS_NOEXCEPT {
  return Upper();
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator double() const MANTIS_NOEXCEPT {
  return Upper();
}

template <typename Real>
constexpr DoubleMantissa<Real>::operator long double() const MANTIS_NOEXCEPT {
  return Upper();
}

template <typename Real>
BinaryNotation DoubleMantissa<Real>::ToBinary(int num_digits) const {
  DoubleMantissa<Real> value = *this;

  BinaryNotation rep;

  // Testing the negation of negativity also functions for NaN.
  rep.positive = !(value < DoubleMantissa<Real>());

  value = Abs(value);
  if (value.Upper() == Real(0) && value.Lower() == Real(0)) {
    rep.exponent = 0;
    rep.digits.resize(num_digits);
    return rep;
  }
  if (std::isinf(*this)) {
    rep.exponent = 0;
    rep.digits = std::vector<unsigned char>{'i', 'n', 'f'};
    return rep;
  }
  if (std::isnan(*this)) {
    rep.exponent = 0;
    rep.digits = std::vector<unsigned char>{'n', 'a', 'n'};
    return rep;
  }

  rep.exponent = std::floor(std::log2(value.Upper())) + 1;

  value = LoadExponent(value, -rep.exponent + 1);
  if (value >= DoubleMantissa<Real>(2)) {
    ++rep.exponent;
    value /= Real(2);
  } else if (value < DoubleMantissa<Real>(1)) {
    --rep.exponent;
    value *= Real(2);
  }

  // We round the very last digit and floor the rest.
  rep.digits.resize(num_digits);
  for (int digit = 0; digit < num_digits - 1; ++digit) {
    // TODO(Jack Poulson): Assert floored_value >= 0 && floored_value < 2.
    const int floored_value = std::floor(value.Upper());
    rep.digits[digit] = floored_value;
    value -= Real(floored_value);
    value *= Real(2);
  }
  if (num_digits > 0) {
    // TODO(Jack Poulson): Assert rounded_value >= 0 && rounded_value <= 2.
    const DoubleMantissa<Real> rounded_value = Round(value);
    const int rounded_value_int = int(rounded_value);
    rep.digits[num_digits - 1] = rounded_value_int;
  }

  // Handle any carries.
  for (int digit = num_digits - 1; digit > 0; --digit) {
    if (rep.digits[digit] > 1) {
      // TODO(Jack Poulson): Assert that digit == 2.
      ++rep.digits[digit - 1];
      rep.digits[digit] = 0;
    }
  }
  if (num_digits > 0 && rep.digits[0] > 1) {
    // Shift all values right one and set the first two to 1 and 0.
    ++rep.exponent;
    for (int digit = num_digits - 1; digit > 0; --digit) {
      rep.digits[digit] = rep.digits[digit - 1];
    }
    if (num_digits > 1) {
      rep.digits[1] = 0;
    }
    rep.digits[0] = 1;
  }

  return rep;
}

template <typename Real>
DecimalNotation DoubleMantissa<Real>::ToDecimal(int num_digits) const {
  DoubleMantissa<Real> value = *this;

  DecimalNotation rep;

  // Testing the negation of negativity also functions for NaN.
  rep.positive = !(value < DoubleMantissa<Real>());

  value = Abs(value);
  if (value.Upper() == Real(0) && value.Lower() == Real(0)) {
    rep.exponent = 0;
    rep.digits.resize(num_digits);
    return rep;
  }
  if (std::isinf(*this)) {
    rep.exponent = 0;
    rep.digits = std::vector<unsigned char>{'i', 'n', 'f'};
    return rep;
  }
  if (std::isnan(*this)) {
    rep.exponent = 0;
    rep.digits = std::vector<unsigned char>{'n', 'a', 'n'};
    return rep;
  }

  rep.exponent = std::floor(std::log10(value.Upper()));

  value *= IntegerPower(DoubleMantissa<Real>(Real(10)), -rep.exponent);
  if (value >= DoubleMantissa<Real>(10)) {
    ++rep.exponent;
    value /= Real(10);
  } else if (value < DoubleMantissa<Real>(1)) {
    --rep.exponent;
    value *= Real(10);
  }

  // We round the very last digit and floor the rest.
  rep.digits.resize(num_digits);
  for (int digit = 0; digit < num_digits - 1; ++digit) {
    const int floored_value = std::max(0, int(std::floor(value.Upper())));
    rep.digits[digit] = floored_value;
    value -= Real(floored_value);
    value *= Real(10);
  }
  if (num_digits > 0) {
    // TODO(Jack Poulson): Assert rounded_value >= 0 && rounded_value <= 10.
    const DoubleMantissa<Real> rounded_value = Round(value);
    const int rounded_value_int = int(rounded_value);
    rep.digits[num_digits - 1] = rounded_value_int;
  }

  // Handle any carries.
  for (int digit = num_digits - 1; digit > 0; --digit) {
    if (rep.digits[digit] > 9) {
      // TODO(Jack Poulson): Assert that digit == 10.
      ++rep.digits[digit - 1];
      rep.digits[digit] = 0;
    }
  }
  if (num_digits > 0 && rep.digits[0] > 9) {
    // Shift all values right one and set the first two to 1 and 0.
    ++rep.exponent;
    for (int digit = num_digits - 1; digit > 0; --digit) {
      rep.digits[digit] = rep.digits[digit - 1];
    }
    if (num_digits > 1) {
      rep.digits[1] = 0;
    }
    rep.digits[0] = 1;
  }

  return rep;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::FromBinary(
    const BinaryNotation& rep) {
  *this = DoubleMantissa<Real>();

  // Specially handle NaN and +-infinity.
  if (rep.digits.size() == 3 && rep.digits[0] == 'i') {
    *this = std::numeric_limits<DoubleMantissa<Real>>::infinity();
    *this = -*this;
    return *this;
  } else if (rep.digits.size() == 3 && rep.digits[0] == 'n') {
    *this = std::numeric_limits<DoubleMantissa<Real>>::quiet_NaN();
    return *this;
  }

  // Load in the digits, least significant first. At the end of the loop, we
  // should have a value that is in the interval [0, 1).
  for (int digit = rep.digits.size() - 1; digit >= 0; --digit) {
    *this += Real(rep.digits[digit]);
    *this /= Real(2);
  }

  // Incorporate the exponent.
  *this = LoadExponent(*this, rep.exponent);

  // Incorporate the sign.
  if (!rep.positive) {
    *this = -*this;
  }

  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::FromDecimal(
    const DecimalNotation& rep) {
  *this = DoubleMantissa<Real>();

  // Specially handle NaN and +-infinity.
  if (rep.digits.size() == 3 && rep.digits[0] == 'i') {
    *this = std::numeric_limits<DoubleMantissa<Real>>::infinity();
    *this = -*this;
    return *this;
  } else if (rep.digits.size() == 3 && rep.digits[0] == 'n') {
    *this = std::numeric_limits<DoubleMantissa<Real>>::quiet_NaN();
    return *this;
  }

  // Load in the digits, least significant first. At the end of the loop, we
  // should have a value that is either 0 or in the interval [1, 10).
  for (int digit = rep.digits.size() - 1; digit >= 0; --digit) {
    *this += Real(rep.digits[digit]);
    if (digit > 0) {
      *this /= Real(10);
    }
  }

  // Incorporate the exponent.
  *this *= IntegerPower(DoubleMantissa<Real>(Real(10)), rep.exponent);

  // Incorporate the sign.
  if (!rep.positive) {
    *this = -*this;
  }

  return *this;
}

template <typename Real>
template <typename UniformRNG>
DoubleMantissa<Real> DoubleMantissa<Real>::UniformRandom(
    UniformRNG& generator) {
  static constexpr int base_digits = std::numeric_limits<Real>::digits;
  std::uniform_real_distribution<Real> base_dist;

  DoubleMantissa<Real> result;
  result = base_dist(generator);
  result = LoadExponent(result, -base_digits);
  result += base_dist(generator);

  return result;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(
    const DoubleMantissa<Real>& value) {
  *this = Divide(*this, value);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real> QuickTwoSum(
    const Real& larger, const Real& smaller) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(larger + smaller);
  result.Lower() = smaller - (result.Upper() - larger);
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> QuickTwoSum(const DoubleMantissa<Real>& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(x.Upper() + x.Lower());
  result.Lower() = x.Lower() - (result.Upper() - x.Upper());
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoSum(const Real& larger,
                                      const Real& smaller) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(larger + smaller);
  const Real smaller_approx = result.Upper() - larger;
  result.Lower() =
      (larger - (result.Upper() - smaller_approx)) + (smaller - smaller_approx);
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoSum(const DoubleMantissa<Real>& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(x.Upper() + x.Lower());
  const Real smaller_approx = result.Upper() - x.Upper();
  result.Lower() = (x.Upper() - (result.Upper() - smaller_approx)) +
                   (x.Lower() - smaller_approx);
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> QuickTwoDiff(
    const Real& larger, const Real& smaller) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(larger - smaller);
  result.Lower() = (larger - result.Upper()) - smaller;
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> QuickTwoDiff(const DoubleMantissa<Real>& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(x.Upper() - x.Lower());
  result.Lower() = (x.Upper() - result.Upper()) - x.Lower();
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoDiff(const Real& larger,
                                       const Real& smaller) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(larger - smaller);
  const Real smaller_approx = larger - result.Upper();
  result.Lower() =
      (larger - (result.Upper() + smaller_approx)) - (smaller - smaller_approx);
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoDiff(const DoubleMantissa<Real>& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(x.Upper() - x.Lower());
  const Real smaller_approx = x.Upper() - result.Upper();
  result.Lower() = (x.Upper() - (result.Upper() + smaller_approx)) -
                   (x.Lower() - smaller_approx);
  return result;
}

template <typename Real>
DoubleMantissa<Real> TwoProdFMA(const Real& x, const Real& y) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> result(x * y);
  result.Lower() = MultiplySubtract(x, y, result.Upper());
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> Split(const Real& value) MANTIS_NOEXCEPT {
  constexpr int num_digits = std::numeric_limits<Real>::digits;
  constexpr int safe_max = std::numeric_limits<Real>::max();

  constexpr Real kSplitter = (1u << num_digits) + 1;
  constexpr Real kSplitScale = 1u << (num_digits + 1);
  constexpr Real kSplitThreshold = safe_max / kSplitScale;

  DoubleMantissa<Real> result;
  if (value > kSplitThreshold || value < -kSplitThreshold) {
    constexpr Real kSplitInvScale = Real{1} / kSplitScale;
    constexpr Real scaled_value = kSplitInvScale * value;

    constexpr Real temp = kSplitter * value;
    result.Upper() = temp - (temp - value);
    result.Lower() = value - result.Upper();

    result.Upper() *= kSplitScale;
    result.Lower() *= kSplitScale;
  } else {
    constexpr Real temp = kSplitter * value;
    result.Upper() = temp - (temp - value);
    result.Lower() = value - result.Upper();
  }
  return result;
}

inline DoubleMantissa<float> TwoProd(const float& x,
                                     const float& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

inline DoubleMantissa<double> TwoProd(const double& x,
                                      const double& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

inline DoubleMantissa<long double> TwoProd(
    const long double& x, const long double& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

template <typename Real>
DoubleMantissa<Real> TwoProd(const Real& x, const Real& y) MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> x_split = Split(x);
  const DoubleMantissa<Real> y_split = Split(y);

  DoubleMantissa<Real> result(x * y);
  result.Lower() =
      ((x_split.Upper() * y_split.Upper() - result.Upper()) +
       x_split.Upper() * y_split.Lower() + x_split.Lower() * y_split.Upper()) +
      x_split.Lower() * y_split.Lower();

  return result;
}

inline DoubleMantissa<float> TwoSquare(const float& x) MANTIS_NOEXCEPT {
  DoubleMantissa<float> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

inline DoubleMantissa<double> TwoSquare(const double& x) MANTIS_NOEXCEPT {
  DoubleMantissa<double> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

inline DoubleMantissa<long double> TwoSquare(const long double& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<long double> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

template <typename Real>
DoubleMantissa<Real> TwoSquare(const Real& x) MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> x_split = Split(x);

  DoubleMantissa<Real> result(x * x);
  result.Lower() = ((x.Upper() * x.Upper() - result.Upper()) +
                    Real(2) * x_split.Upper() * x_split.Lower()) +
                   x_split.Lower() * x_split.Lower();

  return result;
}

template <typename Real>
DoubleMantissa<Real> FastDivide(const DoubleMantissa<Real>& x,
                                const DoubleMantissa<Real>& y) {
  // See the QD library by Hida et al. for a similar implementation.
  DoubleMantissa<Real> quotient(x.Upper() / y.Upper());

  // Compute the remainder, remander := *this - approx_quotient * value.
  const DoubleMantissa<Real> product = y * quotient;
  DoubleMantissa<Real> remainder = TwoDiff(x.Upper(), product.Upper());
  remainder.Lower() += x.Lower();
  remainder.Lower() -= product.Lower();

  // Perform iterative refinement.
  quotient.Lower() = (remainder.Upper() + remainder.Lower()) / y.Upper();

  return QuickTwoSum(quotient);
}

template <typename Real>
DoubleMantissa<Real> Divide(const DoubleMantissa<Real>& x,
                            const DoubleMantissa<Real>& y) {
  // See the QD library by Hida et al. for a similar implementation.
  DoubleMantissa<Real> quotient(x.Upper() / y.Upper());

  // Compute the original remainder.
  DoubleMantissa<Real> r = x - quotient.Upper() * y;

  // Perform a step of iterative refinement.
  quotient.Lower() = r.Upper() / y.Upper();
  r -= quotient.Lower() * y;

  // Perform another step of iterative refinement.
  const Real update = r.Upper() / y.Upper();

  // Combine the original approximation and the two updates into the result.
  quotient = QuickTwoSum(quotient);
  quotient += update;

  return quotient;
}

template <typename Real>
constexpr bool operator==(const DoubleMantissa<Real>& lhs,
                          int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs && lhs.Lower() == Real();
}

template <typename Real>
constexpr bool operator==(const DoubleMantissa<Real>& lhs,
                          long int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs && lhs.Lower() == Real();
}

template <typename Real>
constexpr bool operator==(const DoubleMantissa<Real>& lhs,
                          long long int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs && lhs.Lower() == Real();
}

template <typename Real>
constexpr bool operator==(const DoubleMantissa<Real>& lhs,
                          const Real& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs && lhs.Lower() == Real();
}

template <typename Real>
constexpr bool operator==(int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs == lhs;
}

template <typename Real>
constexpr bool operator==(long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs == lhs;
}

template <typename Real>
constexpr bool operator==(long long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs == lhs;
}

template <typename Real>
constexpr bool operator==(const Real& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs == lhs;
}

template <typename Real>
constexpr bool operator==(const DoubleMantissa<Real>& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs.Upper() && lhs.Lower() == rhs.Lower();
}

template <typename Real>
constexpr bool operator!=(const DoubleMantissa<Real>& lhs,
                          int rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator!=(const DoubleMantissa<Real>& lhs,
                          long int rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator!=(const DoubleMantissa<Real>& lhs,
                          long long int rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator!=(const DoubleMantissa<Real>& lhs,
                          const Real& rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator!=(int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(rhs == lhs);
}

template <typename Real>
constexpr bool operator!=(long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(rhs == lhs);
}

template <typename Real>
constexpr bool operator!=(long long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(rhs == lhs);
}

template <typename Real>
constexpr bool operator!=(const Real& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(rhs == lhs);
}

template <typename Real>
constexpr bool operator!=(const DoubleMantissa<Real>& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator<(const DoubleMantissa<Real>& lhs,
                         int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs || (lhs.Upper() == rhs && lhs.Lower() < Real());
}

template <typename Real>
constexpr bool operator<(const DoubleMantissa<Real>& lhs,
                         long int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs || (lhs.Upper() == rhs && lhs.Lower() < Real());
}

template <typename Real>
constexpr bool operator<(const DoubleMantissa<Real>& lhs,
                         long long int rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs || (lhs.Upper() == rhs && lhs.Lower() < Real());
}

template <typename Real>
constexpr bool operator<(const DoubleMantissa<Real>& lhs,
                         const Real& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs || (lhs.Upper() == rhs && lhs.Lower() < Real());
}

template <typename Real>
constexpr bool operator<(int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs < rhs.Upper() || (lhs == rhs.Upper() && Real() < rhs.Lower());
}

template <typename Real>
constexpr bool operator<(long int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs < rhs.Upper() || (lhs == rhs.Upper() && Real() < rhs.Lower());
}

template <typename Real>
constexpr bool operator<(long long int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs < rhs.Upper() || (lhs == rhs.Upper() && Real() < rhs.Lower());
}

template <typename Real>
constexpr bool operator<(const Real& lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs < rhs.Upper() || (lhs == rhs.Upper() && Real() < rhs.Lower());
}

template <typename Real>
constexpr bool operator<(const DoubleMantissa<Real>& lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs.Upper() ||
         (lhs.Upper() == rhs.Upper() && lhs.Lower() < rhs.Lower());
}

template <typename Real>
constexpr bool operator<=(const DoubleMantissa<Real>& lhs,
                          int rhs) MANTIS_NOEXCEPT {
  return !(rhs < lhs);
}

template <typename Real>
constexpr bool operator<=(const DoubleMantissa<Real>& lhs,
                          long int rhs) MANTIS_NOEXCEPT {
  return !(rhs < lhs);
}

template <typename Real>
constexpr bool operator<=(const DoubleMantissa<Real>& lhs,
                          long long int rhs) MANTIS_NOEXCEPT {
  return !(rhs < lhs);
}

template <typename Real>
constexpr bool operator<=(const DoubleMantissa<Real>& lhs,
                          const Real& rhs) MANTIS_NOEXCEPT {
  return !(rhs < lhs);
}

template <typename Real>
constexpr bool operator<=(const DoubleMantissa<Real>& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(rhs < lhs);
}

template <typename Real>
constexpr bool operator>(int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(long int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(long long int lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const Real& lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const DoubleMantissa<Real>& lhs,
                         int rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const DoubleMantissa<Real>& lhs,
                         long int rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const DoubleMantissa<Real>& lhs,
                         long long int rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const DoubleMantissa<Real>& lhs,
                         const Real& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>(const DoubleMantissa<Real>& lhs,
                         const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return rhs < lhs;
}

template <typename Real>
constexpr bool operator>=(int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(long long int lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const Real& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const DoubleMantissa<Real>& lhs,
                          int rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const DoubleMantissa<Real>& lhs,
                          long int rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const DoubleMantissa<Real>& lhs,
                          long long int rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const DoubleMantissa<Real>& lhs,
                          const Real& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr bool operator>=(const DoubleMantissa<Real>& lhs,
                          const DoubleMantissa<Real>& rhs) MANTIS_NOEXCEPT {
  return !(lhs < rhs);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& value)
    MANTIS_NOEXCEPT {
  return DoubleMantissa<Real>(-value.Upper(), -value.Lower());
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(int x, const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  return Real(x) + y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(
    long int x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  return Real(x) + y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(
    long long int x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  return Real(x) + y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(
    const Real& x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(y);
  z += x;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(const DoubleMantissa<Real>& x,
                                         int y) MANTIS_NOEXCEPT {
  return x + Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(const DoubleMantissa<Real>& x,
                                         long int y) MANTIS_NOEXCEPT {
  return x + Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(const DoubleMantissa<Real>& x,
                                         long long int y) MANTIS_NOEXCEPT {
  return x + Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(const DoubleMantissa<Real>& x,
                                         const Real& y) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z += y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator+(const DoubleMantissa<Real>& x,
                                         const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z += y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(int x, const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  return Real(x) - y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(
    long int x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  return Real(x) - y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(
    long long int x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  return Real(x) - y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(
    const Real& x, const DoubleMantissa<Real>& y) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(-y);
  z += x;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& x,
                                         int y) MANTIS_NOEXCEPT {
  return x - Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& x,
                                         long int y) MANTIS_NOEXCEPT {
  return x - Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& x,
                                         long long int y) MANTIS_NOEXCEPT {
  return x - Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& x,
                                         const Real& y) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z -= y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator-(const DoubleMantissa<Real>& x,
                                         const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z -= y;
  return z;
}

template <typename Real>
DoubleMantissa<Real> operator*(int x,
                               const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  return Real(x) * y;
}

template <typename Real>
DoubleMantissa<Real> operator*(long int x,
                               const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  return Real(x) * y;
}

template <typename Real>
DoubleMantissa<Real> operator*(long long int x,
                               const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  return Real(x) * y;
}

template <typename Real>
DoubleMantissa<Real> operator*(const Real& x,
                               const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(y);
  z *= x;
  return z;
}

template <typename Real>
DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                               int y)MANTIS_NOEXCEPT {
  return x * Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                               long int y)MANTIS_NOEXCEPT {
  return x * Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                               long long int y)MANTIS_NOEXCEPT {
  return x * Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                               const Real& y)MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                               const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
DoubleMantissa<Real> operator/(int x, const DoubleMantissa<Real>& y) {
  return Real(x) / y;
}

template <typename Real>
DoubleMantissa<Real> operator/(long int x, const DoubleMantissa<Real>& y) {
  return Real(x) / y;
}

template <typename Real>
DoubleMantissa<Real> operator/(long long int x, const DoubleMantissa<Real>& y) {
  return Real(x) / y;
}

template <typename Real>
DoubleMantissa<Real> operator/(const Real& x, const DoubleMantissa<Real>& y) {
  return Divide(DoubleMantissa<Real>(x), y);
}

template <typename Real>
DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x, int y) {
  return x / Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x, long int y) {
  return x / Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x, long long int y) {
  return x / Real(y);
}

template <typename Real>
DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x, const Real& y) {
  DoubleMantissa<Real> z(x);
  z /= y;
  return z;
}

template <typename Real>
DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x,
                               const DoubleMantissa<Real>& y) {
  return Divide(x, y);
}

inline float Floor(const float& value) MANTIS_NOEXCEPT {
  return std::floor(value);
}

inline double Floor(const double& value) MANTIS_NOEXCEPT {
  return std::floor(value);
}

inline long double Floor(const long double& value) MANTIS_NOEXCEPT {
  return std::floor(value);
}

template <typename Real>
constexpr DoubleMantissa<Real> Floor(const DoubleMantissa<Real>& value)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> floored_value(std::floor(value.Upper()));
  if (floored_value.Upper() == value.Upper()) {
    // The upper component is already floored to an integer, so floor the
    // lower component.
    floored_value.Lower() = std::floor(value.Lower());
    floored_value.Reduce();
  }
  return floored_value;
}

inline float Round(const float& value) MANTIS_NOEXCEPT {
  return std::round(value);
}

inline double Round(const double& value) MANTIS_NOEXCEPT {
  return std::round(value);
}

inline long double Round(const long double& value) MANTIS_NOEXCEPT {
  return std::round(value);
}

template <typename Real>
constexpr DoubleMantissa<Real> Round(const DoubleMantissa<Real>& value)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> rounded_value(std::round(value.Upper()));
  if (rounded_value.Upper() == value.Upper()) {
    // The upper component is already rounded to an integer, so round the
    // lower component.
    rounded_value.Lower() = std::round(value.Lower());
    rounded_value.Reduce();
  } else if (std::abs(rounded_value.Upper() - value.Upper()) == Real(0.5)) {
    // Resolve the tie based on the low word, which we know must round to
    // zero.
    // NOTE: It appears that QD incorrectly rounds ties when the upper word
    // is negative.
    if (value.Upper() > 0 && value.Lower() < Real(0)) {
      // Values such as 2.5 - eps, eps > 0, should round to 2 rather than 3.
      rounded_value.Upper() -= Real(1);
    } else if (value.Upper() < 0 && value.Lower() > Real(0)) {
      // Values such as -2.5 + eps, eps > 0< should round to -2 rather than
      // -3.
      rounded_value.Upper() += Real(1);
    }
  }
  return rounded_value;
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const DoubleMantissa<Real>& value) {
  constexpr int max_digits10 =
      std::numeric_limits<DoubleMantissa<Real>>::max_digits10;
  const DecimalNotation rep = value.ToDecimal(max_digits10);
  out << rep.ToString();
  return out;
}

}  // namespace mantis

namespace std {

template <typename Real>
string to_string(const mantis::DoubleMantissa<Real>& value) {
  constexpr int max_digits10 =
      numeric_limits<mantis::DoubleMantissa<Real>>::max_digits10;
  const mantis::DecimalNotation rep = value.ToDecimal(max_digits10);
  return rep.ToString();
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_CLASS_IMPL_H_
