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
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(int value)
    MANTIS_NOEXCEPT {
  return *this *= Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(
    const Real& value) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> product = TwoProd(Upper(), value);
  product.Lower() += Lower() * value;
  *this = QuickTwoSum(product);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(
    const DoubleMantissa<Real>& value) MANTIS_NOEXCEPT {
  DoubleMantissa<Real> product = TwoProd(Upper(), value.Upper());
  product.Lower() += value.Lower() * Upper();
  product.Lower() += value.Upper() * Lower();
  *this = QuickTwoSum(product);
  return *this;
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(int value) {
  return *this /= Real(value);
}

template <typename Real>
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(
    const Real& value) {
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
    // TODO(Jack Poulson): Assert floored_value >= 0 && floored_value < 10.
    const int floored_value = std::floor(value.Upper());
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
constexpr DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(
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
constexpr DoubleMantissa<Real> TwoProdFMA(const Real& x,
                                          const Real& y) MANTIS_NOEXCEPT {
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

constexpr DoubleMantissa<float> TwoProd(const float& x,
                                        const float& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

constexpr DoubleMantissa<double> TwoProd(const double& x,
                                         const double& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

constexpr DoubleMantissa<long double> TwoProd(
    const long double& x, const long double& y) MANTIS_NOEXCEPT {
  return TwoProdFMA(x, y);
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoProd(const Real& x,
                                       const Real& y) MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> x_split = Split(x);
  const DoubleMantissa<Real> y_split = Split(y);

  DoubleMantissa<Real> result(x * y);
  result.Lower() =
      ((x_split.Upper() * y_split.Upper() - result.Upper()) +
       x_split.Upper() * y_split.Lower() + x_split.Lower() * y_split.Upper()) +
      x_split.Lower() * y_split.Lower();

  return result;
}

constexpr DoubleMantissa<float> TwoSquare(const float& x) MANTIS_NOEXCEPT {
  DoubleMantissa<float> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

constexpr DoubleMantissa<double> TwoSquare(const double& x) MANTIS_NOEXCEPT {
  DoubleMantissa<double> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

constexpr DoubleMantissa<long double> TwoSquare(const long double& x)
    MANTIS_NOEXCEPT {
  DoubleMantissa<long double> result(x * x);
  result.Lower() = MultiplySubtract(x, x, result.Upper());
  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> TwoSquare(const Real& x) MANTIS_NOEXCEPT {
  const DoubleMantissa<Real> x_split = Split(x);

  DoubleMantissa<Real> result(x * x);
  result.Lower() = ((x.Upper() * x.Upper() - result.Upper()) +
                    Real(2) * x_split.Upper() * x_split.Lower()) +
                   x_split.Lower() * x_split.Lower();

  return result;
}

template <typename Real>
constexpr DoubleMantissa<Real> FastDivide(const DoubleMantissa<Real>& x,
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
constexpr DoubleMantissa<Real> Divide(const DoubleMantissa<Real>& x,
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
                          const Real& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() == rhs && lhs.Lower() == Real();
}

template <typename Real>
constexpr bool operator==(int lhs,
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
                          const Real& rhs) MANTIS_NOEXCEPT {
  return !(lhs == rhs);
}

template <typename Real>
constexpr bool operator!=(int lhs,
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
                         const Real& rhs) MANTIS_NOEXCEPT {
  return lhs.Upper() < rhs || (lhs.Upper() == rhs && lhs.Lower() < Real());
}

template <typename Real>
constexpr bool operator<(int lhs,
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
constexpr DoubleMantissa<Real> operator*(int x, const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  return Real(x) * y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator*(
    const Real& x, const DoubleMantissa<Real>& y)MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(y);
  z *= x;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                                         int y)MANTIS_NOEXCEPT {
  return x * Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                                         const Real& y)MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator*(const DoubleMantissa<Real>& x,
                                         const DoubleMantissa<Real>& y)
    MANTIS_NOEXCEPT {
  DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator/(int x, const DoubleMantissa<Real>& y) {
  return Real(x) / y;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator/(const Real& x,
                                         const DoubleMantissa<Real>& y) {
  return Divide(DoubleMantissa<Real>(x), y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x, int y) {
  return x / Real(y);
}

template <typename Real>
constexpr DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x,
                                         const Real& y) {
  DoubleMantissa<Real> z(x);
  z /= y;
  return z;
}

template <typename Real>
constexpr DoubleMantissa<Real> operator/(const DoubleMantissa<Real>& x,
                                         const DoubleMantissa<Real>& y) {
  return Divide(x, y);
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
DoubleMantissa<Real> Hypot(const DoubleMantissa<Real>& x,
                           const DoubleMantissa<Real>& y) {
  const DoubleMantissa<Real> x_abs = Abs(x);
  const DoubleMantissa<Real> y_abs = Abs(y);

  const DoubleMantissa<Real>& a = y_abs > x_abs ? x : y;
  const DoubleMantissa<Real>& b = y_abs > x_abs ? y : x;
  const DoubleMantissa<Real>& a_abs = y_abs > x_abs ? x_abs : y_abs;

  if (a.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  const DoubleMantissa<Real> t = b / a;
  return a_abs * SquareRoot(Real(1) + Square(t));
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const DoubleMantissa<Real>& value) {
  constexpr int max_digits10 =
      std::numeric_limits<DoubleMantissa<Real>>::max_digits10;
  const DecimalNotation rep = value.ToDecimal(max_digits10);
  out << rep.ToString();
  return out;
}

template <typename Real, typename>
Real LogMax() {
  static const Real log_max = std::log(std::numeric_limits<Real>::max());
  return log_max;
}

template <typename Real, typename>
Real LogOf2() {
  static const Real log_of_2 = std::log(Real{2});
  return log_of_2;
}

template <typename Real, typename>
Real LogOf10() {
  static const Real log_of_10 = std::log(Real{10});
  return log_of_10;
}

template <typename Real, typename>
Real Pi() {
  static const Real pi = std::acos(Real{-1});
  return pi;
}

template <typename Real, typename>
Real EulerNumber() {
  static const Real e = std::exp(Real{0});
  return e;
}

template <typename Real>
DoubleMantissa<Real> Inverse(const DoubleMantissa<Real>& value) {
  return Real(1) / value;
}

template <typename Real>
DoubleMantissa<Real> Square(const Real& value) {
  return TwoSquare(value);
}

template <typename Real>
DoubleMantissa<Real> Square(const DoubleMantissa<Real>& value) {
  DoubleMantissa<Real> product = TwoSquare(value.Upper());
  product.Lower() += Real{2} * value.Lower() * value.Upper();
  return QuickTwoSum(product);
}

template <typename Real>
DoubleMantissa<Real> SquareRoot(const DoubleMantissa<Real>& value) {
  // We make use of Karp and Markstein's square-root algorithm from:
  //
  //   Alan H. Karp and Peter Markstein,
  //   "High Precision Division and Square Root",
  //   ACM TOMS, 23(4):561--589, 1997.
  //
  // As described in said manuscript, the iteration is:
  //
  //   x_{k + 1} = x_k + x_k (1 - a x_k^2) / 2.
  //
  // As described in the QD library of Hida et al., given an initial
  // approximation 'x' of 1 / sqrt(a) -- though the comments in the source code
  // contain a typo suggesting 'x' approximates sqrt(a) -- the square-root can
  // be approximated as:
  //
  //   a * x + [a - (a * x)^2] * (x / 2).
  //
  // While the QD library only performs one iteration, testing immediately
  // revealed factor of four improvements in accuracy from a second iteration.
  // We therefore do so by default and accept the performance penalty.
  if (value.Upper() == Real{0} && value.Lower() == Real{0}) {
    return value;
  }

  if (value.Upper() < Real{0}) {
    return double_mantissa::QuietNan<Real>();
  }

  DoubleMantissa<Real> iterate;
  {
    const Real inv_sqrt = Real{1} / std::sqrt(value.Upper());
    const Real left_term = value.Upper() * inv_sqrt;
    const Real right_term =
        (value - Square(left_term)).Upper() * (inv_sqrt / Real{2});
    iterate = DoubleMantissa<Real>(left_term, right_term).Reduce();
  }

  const bool two_iterations = true;
  if (two_iterations) {
    iterate = DoubleMantissa<Real>(1) / iterate;
    const Real left_term = value.Upper() * iterate.Upper();
    const Real right_term =
        (value - Square(left_term)).Upper() * (iterate.Upper() / Real{2});
    iterate = DoubleMantissa<Real>(left_term, right_term).Reduce();
  }

  return iterate;
}

template <typename Real>
DoubleMantissa<Real> MultiplyByPowerOfTwo(const DoubleMantissa<Real>& value,
                                          const Real& power_of_two) {
  return DoubleMantissa<Real>(value.Upper() * power_of_two,
                              value.Lower() * power_of_two);
}

template <typename Real>
DoubleMantissa<Real> LoadExponent(const DoubleMantissa<Real>& value, int exp) {
  return DoubleMantissa<Real>(std::ldexp(value.Upper(), exp),
                              std::ldexp(value.Lower(), exp));
}

template <typename Real>
DoubleMantissa<Real> Exp(const DoubleMantissa<Real>& value) {
  // We roughly reproduce the range provided within the QD library of Hida et
  // al., where the maximum double-precision exponent is set to 709 despite
  // std::log(numeric_limits<double>::max()) = 709.783.
  static const Real log_max = LogMax<Real>();
  static const Real exp_max = log_max - Real{0.783};

  if (value.Upper() <= -exp_max) {
    // Underflow to zero.
    return DoubleMantissa<Real>();
  }
  if (value.Upper() >= exp_max) {
    // Overflow to infinity.
    return double_mantissa::Infinity<Real>();
  }
  if (value.Upper() == Real{0}) {
    return DoubleMantissa<Real>(Real{1});
  }

  // TODO(Jack Poulson): Consider returning a fully-precise result for
  // e = exp(1). At the moment, we compute this constant by calling this
  // routine.

  static const DoubleMantissa<Real> log_of_2 = double_mantissa::LogOf2<Real>();
  const Real shift = std::floor(value.Upper() / log_of_2.Upper() + Real{0.5});

  static constexpr int digits =
      std::numeric_limits<DoubleMantissa<Real>>::digits;

  // TODO(Jack Poulson): Determine the number of squarings based on precision.
  const unsigned int num_squares = 9;

  // TODO(Jack Poulson): Calculate a tighter upper bound on the number of terms.
  // Cf. MPFR exponential discussion.
  static constexpr int num_terms = digits / num_squares;

  const Real scale = 1u << num_squares;
  const Real inv_scale = Real{1} / scale;

  DoubleMantissa<Real> r =
      MultiplyByPowerOfTwo(value - log_of_2 * shift, inv_scale);

  // Initialize r_power := r^2.
  DoubleMantissa<Real> r_power = Square(r);

  // Initialize iterate := r + (1/2!) r^2.
  DoubleMantissa<Real> iterate = r + MultiplyByPowerOfTwo(r_power, Real{0.5});

  // r_power := r^3.
  r_power *= r;

  // Compute the update term, (1 / 3!) r^3.
  DoubleMantissa<Real> coefficient =
      DoubleMantissa<Real>(1) / DoubleMantissa<Real>(6);
  DoubleMantissa<Real> term = coefficient * r_power;

  // iterate := r + (1 / 2!) r^2 + (1 / 3!) r^3.
  iterate += term;

  const Real tolerance =
      std::numeric_limits<DoubleMantissa<Real>>::epsilon().Upper() * inv_scale;
  for (int j = 4; j < num_terms; ++j) {
    // (r_power)_j = r^j.
    r_power *= r;

    // coefficient_j = 1 / j!.
    coefficient /= Real(j);

    // term_j = (1 / j!) r^j.
    term = coefficient * r_power;

    // iterate_j := r + (1/2!) r^2 + ... + (1/j!) r^j.
    iterate += term;

    // Early-exit if the terms in the series have become sufficiently small.
    if (std::abs(term.Upper()) <= tolerance) {
      break;
    }
  }

  // Convert z = exp(r) - 1 into exp(r)^{2^{num_squares}} - 1.
  for (unsigned int j = 0; j < num_squares; ++j) {
    // Given that our value represents the component of the exponential without
    // the '1' term in the Taylor series, we recognize that
    //
    //   (x -1)^2 + 2 (x - 1) = x^2 - 1,
    //
    // so that our update formula squares while preserving the shift.
    iterate = Square(iterate) + MultiplyByPowerOfTwo(iterate, Real{2});
  }

  // We are done rescaling, so add the first term of the series in.
  iterate += Real{1};

  // Return 2^{shift} exp(r)^{2^{num_squares}}.
  iterate = LoadExponent(iterate, static_cast<int>(shift));

  return iterate;
}

template <typename Real>
DoubleMantissa<Real> IntegerPower(const DoubleMantissa<Real>& value,
                                  int exponent) {
  if (exponent == 0) {
    if (value == DoubleMantissa<Real>(Real(0))) {
      // We insist that 0^0 is NaN.
      return double_mantissa::QuietNan<Real>();
    }
    return DoubleMantissa<Real>(Real(1));
  } else if (exponent == 1) {
    return value;
  } else if (exponent == -1) {
    return Real(1) / value;
  }

  // Run binary exponentiation with the absolute-value of the exponent.
  DoubleMantissa<Real> scale = value;
  DoubleMantissa<Real> product(Real(1));
  int exponent_abs = std::abs(exponent);
  while (exponent_abs > 0) {
    if (exponent_abs % 2) {
      product *= scale;
    }
    exponent_abs /= 2;
    if (exponent_abs > 0) {
      scale = Square(scale);
    }
  }

  // Invert the product if the exponent was negative.
  if (exponent < 0) {
    product = Real(1) / product;
  }

  return product;
}

template <typename Real>
DoubleMantissa<Real> Power(const DoubleMantissa<Real>& value,
                           const DoubleMantissa<Real>& exponent) {
  if (exponent == std::floor(exponent)) {
    // TODO(Jack Poulson): Make sure there is no over/underflow.
    const int exponent_int = int(exponent);
    return IntegerPower(value, exponent_int);
  }

  return Exp(exponent * Log(value));
}

template <typename Real>
DoubleMantissa<Real> Log(const DoubleMantissa<Real>& value) {
  if (value.Upper() <= Real{0}) {
    return double_mantissa::QuietNan<Real>();
  }
  if (value.Upper() == Real{1} && value.Lower() == Real{0}) {
    return DoubleMantissa<Real>();
  }

  // As described by Hida et al., the basic setup is Newton's algorithm applied
  // to the function f(x) = exp(x) - a. The result is an algorithm
  //
  //   x_{k + 1} = x_k - Df_{x_k}^{-1} f(x_k)
  //             = x_k - (exp(x_k))^{-1} (exp(x_k) - a)
  //             = x_k - (1 - a exp(-x_k))
  //             = x_k + a exp(-x_k) - 1.
  //
  // It is argued that only one iteration is required due to the quadratic
  // convergence properties of Newton's algorithm -- with the understanding
  // that the original iterate is within the basin of convergence and accurate
  // to roughly the precision of the single-mantissa representation.
  //
  // But, especially for large arguments, it has been observed that adding an
  // additional iteration leads to about an order of magnitude increase in
  // accuracy. We therefore accept the performance penalty for this extra
  // digit of confidence.
  DoubleMantissa<Real> x = std::log(value.Upper());

  const int num_iter = 2;
  for (int j = 0; j < num_iter; ++j) {
    // TODO(Jack Poulson): Analyze whether this order of operations is ideal.
    // Hopefully we can avoid catastrophic cancellation.
    x += value * Exp(-x);
    x -= Real{1};
  }

  return x;
}

template <typename Real>
DoubleMantissa<Real> Log2(const DoubleMantissa<Real>& value) {
  return Log(value) / double_mantissa::LogOf2<Real>();
}

template <typename Real>
DoubleMantissa<Real> Log10(const DoubleMantissa<Real>& value) {
  return Log(value) / double_mantissa::LogOf10<Real>();
}

template <typename Real>
DoubleMantissa<Real> Abs(const DoubleMantissa<Real>& value) {
  return value.Upper() < Real(0) ? -value : value;
}

template <typename Real>
DoubleMantissa<Real> SmallArgSin(const DoubleMantissa<Real>& theta) {
  if (theta.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  // The Taylor series is of the form: x - (1/3!) x^3 + (1/5!) x^5 - ...,
  // we the successive powers of 'x' will be multiplied by -x^2.
  const DoubleMantissa<Real> power_ratio = -Square(theta);

  // Initialize x_power := x and the iterate as the first term in the series.
  DoubleMantissa<Real> x_power = theta;
  DoubleMantissa<Real> iterate(theta);

  const Real threshold = Real(0.5) * std::abs(theta.Upper()) *
                         double_mantissa::Epsilon<Real>().Upper();

  DoubleMantissa<Real> coefficient = 1;
  for (int j = 3;; j += 2) {
    // Update to the next term's power of x.
    x_power *= power_ratio;

    // Update the coefficient to that of the next term.
    coefficient /= Real(j * (j - 1));

    // Store the next term in the Taylor series.
    const DoubleMantissa<Real> term = coefficient * x_power;

    // Add the new term to the iterate.
    iterate += term;

    // If the terms have become sufficiently small, stop the iteration.
    if (std::abs(term.Upper()) <= threshold) {
      break;
    }
  }

  return iterate;
}

template <typename Real>
DoubleMantissa<Real> SmallArgCos(const DoubleMantissa<Real>& theta) {
  if (theta.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  // The Taylor series is of the form: 1 - (1/2!) x^2 + (1/4!) x^4 - ...,
  // we the successive powers of 'x' will be multiplied by -x^2.
  const DoubleMantissa<Real> power_ratio = -Square(theta);

  // Initialize the series at 1 - (1/2!) x^2.
  DoubleMantissa<Real> x_power = power_ratio;
  DoubleMantissa<Real> coefficient = Real(0.5);
  DoubleMantissa<Real> iterate =
      DoubleMantissa<Real>(1) + MultiplyByPowerOfTwo(x_power, Real(0.5));

  const Real threshold = Real(0.5) * double_mantissa::Epsilon<Real>().Upper();

  DoubleMantissa<Real> denominator(Real(1));
  for (int j = 4;; j += 2) {
    // Update to the next term's power of x.
    x_power *= power_ratio;

    // Update the coefficient to that of the next term.
    coefficient /= Real(j * (j - 1));

    // Store the next term in the Taylor series.
    const DoubleMantissa<Real> term = coefficient * x_power;

    // Add the new term to the iterate.
    iterate += term;

    // If the terms have become sufficiently small, stop the iteration.
    if (std::abs(term.Upper()) <= threshold) {
      break;
    }
  }

  return iterate;
}

template <typename Real>
DoubleMantissa<Real> SinCosDualMagnitude(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }
  if (value.Upper() == Real(1) && value.Lower() == Real(0)) {
    return DoubleMantissa<Real>(Real(0));
  }
  return SquareRoot(DoubleMantissa<Real>(Real(1)) - Square(value));
}

template <typename Real>
void DecomposeSinCosArgument(const DoubleMantissa<Real>& theta,
                             int* num_half_pi_int, int* num_sixteenth_pi_int,
                             DoubleMantissa<Real>* sixteenth_pi_remainder) {
  static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
  static const DoubleMantissa<Real> two_pi = Real(2) * pi;
  const DoubleMantissa<Real> num_two_pi = Round(theta / two_pi);
  const DoubleMantissa<Real> two_pi_remainder = theta - two_pi * num_two_pi;

  static const DoubleMantissa<Real> half_pi = pi / Real(2);
  const Real num_half_pi =
      std::floor(two_pi_remainder.Upper() / half_pi.Upper() + Real(0.5));
  const DoubleMantissa<Real> half_pi_remainder =
      two_pi_remainder - half_pi * num_half_pi;
  *num_half_pi_int = num_half_pi;
  if (*num_half_pi_int < -2 || *num_half_pi_int > 2) {
#ifdef MANTIS_DEBUG
    std::cerr << "Could not find modulus of arg relative to pi / 2."
              << std::endl;
    *sixteenth_pi_remainder = double_mantissa::QuietNan<Real>();
    return;
#endif  // ifdef MANTIS_DEBUG
  }

  static const DoubleMantissa<Real> sixteenth_pi = pi / Real(16);
  const Real num_sixteenth_pi =
      std::floor(half_pi_remainder.Upper() / sixteenth_pi.Upper() + Real(0.5));
  *sixteenth_pi_remainder = half_pi_remainder - sixteenth_pi * num_sixteenth_pi;
  *num_sixteenth_pi_int = num_sixteenth_pi;
  const int abs_num_sixteenth_pi_int = std::abs(*num_sixteenth_pi_int);
  if (abs_num_sixteenth_pi_int > 4) {
#ifdef MANTIS_DEBUG
    std::cerr << "Could not find modulus of arg relative to pi / 16."
              << std::endl;
    *sixteenth_pi_remainder = double_mantissa::QuietNan<Real>();
#endif  // ifdef MANTIS_DEBUG
  }
}

template <typename Real>
const DoubleMantissa<Real>& SixteenthPiSinTable(int num_sixteenth_pi) {
#ifdef MANTIS_DEBUG
  if (num_sixteenth_pi < 1 || num_sixteenth_pi > 4) {
    std::cerr << "Only multiples of {1, 2, 3, 4} of pi / 16 supported."
              << std::endl;
    return double_mantissa::QuietNan<Real>();
  }
#endif  // ifdef MANTIS_DEBUG
  static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
  static const DoubleMantissa<Real> sixteenth_pi = pi / Real(16);
  static const DoubleMantissa<Real> sin_table[] = {
      SmallArgSin(sixteenth_pi), SmallArgSin(Real(2) * sixteenth_pi),
      SmallArgSin(Real(3) * sixteenth_pi), SmallArgSin(Real(4) * sixteenth_pi),
  };
  return sin_table[num_sixteenth_pi - 1];
}

template <typename Real>
const DoubleMantissa<Real>& SixteenthPiCosTable(int num_sixteenth_pi) {
#ifdef MANTIS_DEBUG
  if (num_sixteenth_pi < 1 || num_sixteenth_pi > 4) {
    std::cerr << "Only multiples of {1, 2, 3, 4} of pi / 16 supported."
              << std::endl;
    return double_mantissa::QuietNan<Real>();
  }
#endif  // ifdef MANTIS_DEBUG
  static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
  static const DoubleMantissa<Real> sixteenth_pi = pi / Real(16);
  static const DoubleMantissa<Real> cos_table[] = {
      SmallArgCos(sixteenth_pi), SmallArgCos(Real(2) * sixteenth_pi),
      SmallArgCos(Real(3) * sixteenth_pi), SmallArgCos(Real(4) * sixteenth_pi),
  };
  return cos_table[num_sixteenth_pi - 1];
}

template <typename Real>
DoubleMantissa<Real> Sin(const DoubleMantissa<Real>& theta) {
  if (theta.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  // We make thorough usage of sin(a + b) = sin(a) cos(b) + cos(a) sin(b)
  // given a = |num_sixteenth_pi_int * sixteenth_pi| and
  // b = sixteenth_pi_remainder.

  if (num_sixteenth_pi_int == 0) {
    if (num_half_pi_int == 0) {
      // sin(a + b) = sin(b).
      return SmallArgSin(sixteenth_pi_remainder);
    } else if (num_half_pi_int == 1) {
      // sin(a + b + (pi / 2)) = sin(b + (pi / 2)) = cos(b).
      return SmallArgCos(sixteenth_pi_remainder);
    } else if (num_half_pi_int == -1) {
      // sin(a + b - (pi / 2)) = sin(b - (pi / 2)) = -cos(b).
      return -SmallArgCos(sixteenth_pi_remainder);
    } else {
      // sin(a + b + pi) = sin(b + pi) = -sin(b).
      return -SmallArgSin(sixteenth_pi_remainder);
    }
  }

  const DoubleMantissa<Real> sin_a =
      SixteenthPiSinTable<Real>(abs_num_sixteenth_pi_int);
  const DoubleMantissa<Real> cos_a =
      SixteenthPiCosTable<Real>(abs_num_sixteenth_pi_int);

  const DoubleMantissa<Real> sin_b = SmallArgSin(sixteenth_pi_remainder);
  const DoubleMantissa<Real> cos_b = SinCosDualMagnitude(sin_b);

  DoubleMantissa<Real> sin_theta;
  if (num_half_pi_int == 0) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      sin_theta = sin_a * cos_b + cos_a * sin_b;
    } else {
      // sin(-a + b) = sin(-a) cos(b) + cos(-a) sin(b)
      //             = -sin(a) cos(b) + cos(a) sin(b).
      sin_theta = -sin_a * cos_b + cos_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + (pi / 2)) = cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      sin_theta = cos_a * cos_b - sin_a * sin_b;
    } else {
      // sin(-a + b + (pi / 2)) = cos(-a + b)
      //     = cos(a) cos(b) + sin(a) sin(b).
      sin_theta = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b - (pi / 2)) = -cos(a + b)
      //     = -cos(a) cos(b) + sin(a) sin(b).
      sin_theta = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // sin(-a + b - (pi / 2)) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      sin_theta = -cos_a * cos_b - sin_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + pi) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      sin_theta = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // sin(-a + b + pi) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      sin_theta = sin_a * cos_b - cos_a * sin_b;
    }
  }

  return sin_theta;
}

template <typename Real>
DoubleMantissa<Real> Cos(const DoubleMantissa<Real>& theta) {
  if (theta.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  // We make thorough usage of cos(a + b) = cos(a) cos(b) - sin(a) sin(b)
  // given a = |num_sixteenth_pi_int * sixteenth_pi| and
  // b = sixteenth_pi_remainder.

  if (num_sixteenth_pi_int == 0) {
    if (num_half_pi_int == 0) {
      // cos(a + b) = cos(b).
      return SmallArgCos(sixteenth_pi_remainder);
    } else if (num_half_pi_int == 1) {
      // cos(a + b + (pi / 2)) = cos(b + (pi / 2)) = -sin(b).
      return -SmallArgSin(sixteenth_pi_remainder);
    } else if (num_half_pi_int == -1) {
      // cos(a + b - (pi / 2)) = cos(b - (pi / 2)) = sin(b).
      return SmallArgSin(sixteenth_pi_remainder);
    } else {
      // cos(a + b + pi) = cos(b + pi) = -cos(b).
      return -SmallArgCos(sixteenth_pi_remainder);
    }
  }

  const DoubleMantissa<Real> sin_a =
      SixteenthPiSinTable<Real>(abs_num_sixteenth_pi_int);
  const DoubleMantissa<Real> cos_a =
      SixteenthPiCosTable<Real>(abs_num_sixteenth_pi_int);

  const DoubleMantissa<Real> sin_b = SmallArgSin(sixteenth_pi_remainder);
  const DoubleMantissa<Real> cos_b = SinCosDualMagnitude(sin_b);

  DoubleMantissa<Real> cos_theta;
  if (num_half_pi_int == 0) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      cos_theta = cos_a * cos_b - sin_a * sin_b;
    } else {
      // cos(-a + b) = cos(a) cos(b) + sin(a) sin(b).
      cos_theta = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b + (pi / 2)) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      cos_theta = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // cos(-a + b + (pi / 2)) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      cos_theta = sin_a * cos_b - cos_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b - (pi / 2)) = sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      cos_theta = sin_a * cos_b + cos_a * sin_b;
    } else {
      // cos(-a + b - (pi / 2)) = sin(-a + b) = -sin(a) cos(b) + cos(a) sin(b).
      cos_theta = -sin_a * cos_b + cos_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b + pi) = -cos(a + b) = -cos(a) cos(b) + sin(a) sin(b).
      cos_theta = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // cos(-a + b + pi) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      cos_theta = -cos_a * cos_b - sin_a * sin_b;
    }
  }

  return cos_theta;
}

template <typename Real>
void SinCos(const DoubleMantissa<Real>& theta, DoubleMantissa<Real>* sin_theta,
            DoubleMantissa<Real>* cos_theta) {
  if (theta.Upper() == Real(0)) {
    *sin_theta = DoubleMantissa<Real>();
    *cos_theta = DoubleMantissa<Real>(Real(1));
    return;
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  // We make thorough usage of sin(a + b) = sin(a) cos(b) + cos(a) sin(b)
  // given a = |num_sixteenth_pi_int * sixteenth_pi| and
  // b = sixteenth_pi_remainder.

  const DoubleMantissa<Real> sin_b = SmallArgSin(sixteenth_pi_remainder);
  const DoubleMantissa<Real> cos_b = SinCosDualMagnitude(sin_b);

  if (num_sixteenth_pi_int == 0) {
    if (num_half_pi_int == 0) {
      // sin(a + b) = sin(b).
      *sin_theta = sin_b;
      // cos(a + b) = cos(b).
      *cos_theta = cos_b;
    } else if (num_half_pi_int == 1) {
      // sin(a + b + (pi / 2)) = sin(b + (pi / 2)) = cos(b).
      *sin_theta = cos_b;
      // cos(a + b + (pi / 2)) = cos(b + (pi / 2)) = -sin(b).
      *cos_theta = -sin_b;
    } else if (num_half_pi_int == -1) {
      // sin(a + b - (pi / 2)) = sin(b - (pi / 2)) = -cos(b).
      *sin_theta = -cos_b;
      // cos(a + b - (pi / 2)) = cos(b - (pi / 2)) = sin(b).
      *cos_theta = sin_b;
    } else {
      // sin(a + b + pi) = sin(b + pi) = -sin(b).
      *sin_theta = -sin_b;
      // cos(a + b + pi) = cos(b + pi) = -cos(b).
      *cos_theta = -cos_b;
    }
    return;
  }

  const DoubleMantissa<Real> sin_a =
      SixteenthPiSinTable<Real>(abs_num_sixteenth_pi_int);
  const DoubleMantissa<Real> cos_a =
      SixteenthPiCosTable<Real>(abs_num_sixteenth_pi_int);

  if (num_half_pi_int == 0) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      *sin_theta = sin_a * cos_b + cos_a * sin_b;
      // cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      *cos_theta = cos_a * cos_b - sin_a * sin_b;
    } else {
      // sin(-a + b) = sin(-a) cos(b) + cos(-a) sin(b)
      //             = -sin(a) cos(b) + cos(a) sin(b).
      *sin_theta = -sin_a * cos_b + cos_a * sin_b;
      // cos(-a + b) = cos(a) cos(b) + sin(a) sin(b).
      *cos_theta = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + (pi / 2)) = cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      *sin_theta = cos_a * cos_b - sin_a * sin_b;
      // cos(a + b + (pi / 2)) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      *cos_theta = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // sin(-a + b + (pi / 2)) = cos(-a + b)
      //     = cos(a) cos(b) + sin(a) sin(b).
      *sin_theta = cos_a * cos_b + sin_a * sin_b;
      // cos(-a + b + (pi / 2)) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      *cos_theta = sin_a * cos_b - cos_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b - (pi / 2)) = -cos(a + b)
      //     = -cos(a) cos(b) + sin(a) sin(b).
      *sin_theta = -cos_a * cos_b + sin_a * sin_b;
      // cos(a + b - (pi / 2)) = sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      *cos_theta = sin_a * cos_b + cos_a * sin_b;
    } else {
      // sin(-a + b - (pi / 2)) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      *sin_theta = -cos_a * cos_b - sin_a * sin_b;
      // cos(-a + b - (pi / 2)) = sin(-a + b) = -sin(a) cos(b) + cos(a) sin(b).
      *cos_theta = -sin_a * cos_b + cos_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + pi) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      *sin_theta = -sin_a * cos_b - cos_a * sin_b;
      // cos(a + b + pi) = -cos(a + b) = -cos(a) cos(b) + sin(a) sin(b).
      *cos_theta = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // sin(-a + b + pi) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      *sin_theta = sin_a * cos_b - cos_a * sin_b;
      // cos(-a + b + pi) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      *cos_theta = -cos_a * cos_b - sin_a * sin_b;
    }
  }
}

template <typename Real>
DoubleMantissa<Real> Tan(const DoubleMantissa<Real>& theta) {
  DoubleMantissa<Real> sin_theta, cos_theta;
  SinCos(theta, &sin_theta, &cos_theta);
  return sin_theta / cos_theta;
}

template <typename Real>
DoubleMantissa<Real> ArcTan(const DoubleMantissa<Real>& tan_theta) {
  return ArcTan2(tan_theta, DoubleMantissa<Real>(1));
}

template <typename Real>
DoubleMantissa<Real> ArcTan2(const DoubleMantissa<Real>& y,
                             const DoubleMantissa<Real>& x) {
  static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
  static const DoubleMantissa<Real> half_pi =
      MultiplyByPowerOfTwo(pi, Real(0.5));
  static const DoubleMantissa<Real> quarter_pi =
      MultiplyByPowerOfTwo(pi, Real(0.25));
  static const DoubleMantissa<Real> three_quarters_pi = Real(3) * quarter_pi;

  if (x.Upper() == Real(0)) {
    if (y.Upper() == Real(0)) {
      // Every value of theta works equally well for the origin.
      return double_mantissa::QuietNan<Real>();
    }

    // We are on the y axis.
    return y.Upper() > Real(0) ? half_pi : -half_pi;
  } else if (y.Upper() == Real(0)) {
    // We are on the x axis.
    return x.Upper() > Real(0) ? DoubleMantissa<Real>() : pi;
  }

  if (x == y) {
    // We are on the 45 degree line in the upper-right or lower-left quadrant.
    return y.Upper() > Real(0) ? quarter_pi : -three_quarters_pi;
  }

  if (x == -y) {
    // We are on the 135 degree line in the upper-left or lower-right quadrant.
    return y.Upper() > Real(0) ? three_quarters_pi : -quarter_pi;
  }

  // We can directly compute the unique allowable radius, r := || (x, y) ||_2.
  // NOTE: While the QD library of Hida et al. uses the naive formula, we make
  // use of the hypot function -- whose zero branch could be avoided -- to
  // more carefully compute the result.
  const DoubleMantissa<Real> radius = Hypot(x, y);

  // Project our point to the unit circle.
  const DoubleMantissa<Real> x_unit = x / radius;
  const DoubleMantissa<Real> y_unit = y / radius;

  // Initialize with the single-mantissa result.
  DoubleMantissa<Real> theta = std::atan2(y.Upper(), x.Upper());

  // While the QD library uses a single iteration, it is often observed that
  // an additional iteration leads to almost an extra digit of accuracy.
  const int num_iterations = 2;

  DoubleMantissa<Real> sin_theta, cos_theta;
  if (std::abs(x.Upper()) > std::abs(y.Upper())) {
    // Apply Newton's method to the function
    //
    //   f(theta) = sin(theta) - y_unit,
    //
    // which leads to the update
    //
    //   theta_{k + 1} = theta_k - Df_{theta_k}^{-1}(f(theta_k))
    //                 = theta_k - (cos(theta_k))^{-1} (sin(theta_k) - y_unit)
    //                 = theta_k + (y_unit - sin(theta_k)) / cos(theta_k).
    //
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      SinCos(theta, &sin_theta, &cos_theta);
      theta += (y_unit - sin_theta) / cos_theta;
    }
  } else {
    // Apply Newton's method to the function
    //
    //   f(theta) = cos(theta) - x_unit,
    //
    // which leads to the update
    //
    //   theta_{k + 1} = theta_k - Df_{theta_k}^{-1}(f(theta_k))
    //                 = theta_k - (-sin(theta_k))^{-1} (cos(theta_k) -
    //                 x_unit)
    //                 = theta_k - (x_unit - cos(theta_k)) / sin(theta_k).
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      SinCos(theta, &sin_theta, &cos_theta);
      theta -= (x_unit - cos_theta) / sin_theta;
    }
  }

  return theta;
}

template <typename Real>
DoubleMantissa<Real> ArcSin(const DoubleMantissa<Real>& sin_theta) {
  const DoubleMantissa<Real> abs_sin_theta = Abs(sin_theta);
  if (abs_sin_theta.Upper() > Real(1)) {
    return double_mantissa::QuietNan<Real>();
  }
  if (abs_sin_theta.Upper() == Real(1) && abs_sin_theta.Lower() == Real(0)) {
    static const DoubleMantissa<Real> half_pi =
        MultiplyByPowerOfTwo(double_mantissa::Pi<Real>(), Real(0.5));
    return sin_theta.Upper() > 0 ? half_pi : -half_pi;
  }

  const DoubleMantissa<Real> cos_theta = SinCosDualMagnitude(sin_theta);
  return ArcTan2(sin_theta, cos_theta);
}

template <typename Real>
DoubleMantissa<Real> ArcCos(const DoubleMantissa<Real>& cos_theta) {
  const DoubleMantissa<Real> abs_cos_theta = Abs(cos_theta);
  if (abs_cos_theta.Upper() > Real(1)) {
    return double_mantissa::QuietNan<Real>();
  }
  if (abs_cos_theta.Upper() == Real(1) && abs_cos_theta.Lower() == Real(0)) {
    static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
    return cos_theta.Upper() > 0 ? DoubleMantissa<Real>() : pi;
  }

  const DoubleMantissa<Real> sin_theta = SinCosDualMagnitude(cos_theta);
  return ArcTan2(sin_theta, cos_theta);
}

template <typename Real>
DoubleMantissa<Real> HyperbolicSin(const DoubleMantissa<Real>& x) {
  if (x.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(0));
  }

  // We use the same switching point as the QD library of Hida et al.
  const Real kDirectFormulaThresh = Real(0.05);
  if (std::abs(x.Upper()) > kDirectFormulaThresh) {
    const DoubleMantissa<Real> exp_x = Exp(x);
    return MultiplyByPowerOfTwo(exp_x - Inverse(exp_x), Real(0.5));
  }

  // We fall back to a truncation of the Taylor series:
  //
  //   sinh(x) = x + (1 / 3!) x^3 + (1 / 5!) x^5 + ...
  //           = \sum_{n = 0}^{\infty} x^{2 n + 1} / (2 n + 1)!.
  //
  const DoubleMantissa<Real> epsilon =
      std::numeric_limits<DoubleMantissa<Real>>::epsilon();
  const Real term_threshold = std::abs(x.Upper()) * epsilon.Upper();

  // The powers of x increase by x^2 in each term.
  const DoubleMantissa<Real> power_ratio = Square(x);

  // Initialize the partial sum as the first term, x.
  DoubleMantissa<Real> term = x;
  DoubleMantissa<Real> iterate = x;

  for (int m = 3;; m += 2) {
    term *= power_ratio;
    term /= Real((m - 1) * m);
    iterate += term;
    if (std::abs(term.Upper()) <= term_threshold) {
      break;
    }
  }

  return iterate;
}

template <typename Real>
DoubleMantissa<Real> HyperbolicCos(const DoubleMantissa<Real>& x) {
  if (x.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  // Use the defining formula.
  const DoubleMantissa<Real> exp_x = Exp(x);
  return MultiplyByPowerOfTwo(exp_x + Inverse(exp_x), Real(0.5));
}

template <typename Real>
void HyperbolicSinCos(const DoubleMantissa<Real>& x,
                      DoubleMantissa<Real>* sinh_x,
                      DoubleMantissa<Real>* cosh_x) {
  const Real kDirectFormulaThresh = Real(0.05);
  if (std::abs(x.Upper()) <= kDirectFormulaThresh) {
    *sinh_x = HyperbolicSin(x);
    *cosh_x = SquareRoot(Real(1) + Square(*sinh_x));
  } else {
    const DoubleMantissa<Real> exp_x = Exp(x);
    const DoubleMantissa<Real> inv_exp_x = Inverse(exp_x);
    *sinh_x = MultiplyByPowerOfTwo(exp_x - inv_exp_x, Real(0.5));
    *cosh_x = MultiplyByPowerOfTwo(exp_x + inv_exp_x, Real(0.5));
  }
}

template <typename Real>
DoubleMantissa<Real> HyperbolicTan(const DoubleMantissa<Real>& x) {
  if (x.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(0));
  }

  // We use the same switching point as the QD library of Hida et al.
  const Real kDirectFormulaThresh = Real(0.05);
  if (std::abs(x.Upper()) > kDirectFormulaThresh) {
    const DoubleMantissa<Real> exp_x = Exp(x);
    const DoubleMantissa<Real> inv_exp_x = Inverse(exp_x);
    return (exp_x - inv_exp_x) / (exp_x + inv_exp_x);
  }

  // Fall back to a careful calculation of hyperbolic sine.
  const DoubleMantissa<Real> sinh_x = HyperbolicSin(x);

  // Convert the calculation of sinh into cosh using the identity:
  //
  //   cosh^2(x) - sinh^2(x) = 1.
  //
  const DoubleMantissa<Real> cosh_x = SquareRoot(Real(1) + Square(sinh_x));

  return sinh_x / cosh_x;
}

template <typename Real>
DoubleMantissa<Real> ArcHyperbolicSin(const DoubleMantissa<Real>& sinh_x) {
  // Combine the identities:
  //
  //   cosh(x) = sqrt(1 + sinh^2(x)),
  //
  //   cosh(x) + sinh(x) = exp(x),
  //
  // to convert sinh(x) into x.
  const DoubleMantissa<Real> cosh_x = SquareRoot(Real(1) + Square(sinh_x));
  return Log(sinh_x + cosh_x);
}

template <typename Real>
DoubleMantissa<Real> ArcHyperbolicCos(const DoubleMantissa<Real>& cosh_x) {
  if (cosh_x.Upper() < Real(1)) {
    return double_mantissa::QuietNan<Real>();
  }

  // Combine the identities:
  //
  //   sinh(x) = sqrt(cosh^2(x) - 1),
  //
  //   cosh(x) + sinh(x) = exp(x),
  //
  // to convert cosh(x) into x.
  const DoubleMantissa<Real> sinh_x = SquareRoot(Square(cosh_x) - Real(1));
  return Log(sinh_x + cosh_x);
}

template <typename Real>
DoubleMantissa<Real> ArcHyperbolicTan(const DoubleMantissa<Real>& tanh_x) {
  if (std::abs(tanh_x.Upper()) >= Real(1)) {
    return double_mantissa::QuietNan<Real>();
  }

  // tanh(x) = sinh(x) / cosh(x)
  //         =  (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
  //
  // Then
  //
  //   1 + tanh(x) = 2 exp(x) / (exp(x) + exp(-x)),
  //
  //   1 - tanh(x) = 2 exp(-x) / (exp(x) + exp(-x)).
  //
  // Thus,
  //
  //   (1 + tanh(x)) / (1 - tanh(x)) = exp(2 x),
  //
  // yielding the inversion formula:
  //
  //   atanh(x) = (1 / 2) log((1 + x) / (1 - x)).
  //
  return MultiplyByPowerOfTwo(Log((Real(1) + tanh_x) / (Real(1) - tanh_x)),
                              Real(0.5));
}

namespace double_mantissa {

template <typename Real, typename>
constexpr DoubleMantissa<Real> Epsilon() {
  constexpr Real epsilon_single = std::numeric_limits<Real>::epsilon();

  // The 'numeric_limits' definition of epsilon_single is 2^{1 - p} if the
  // binary precision is p digits. Thus, if the precision is doubled, epsilon
  // becomes 2^{1 - 2 p} = (2^{1 - p})^2 / 2 = epsilon_single^2 / 2.
  //
  // This approach is preferable because it allows us to compute the new
  // precision as a constexpr (avoiding pow).
  constexpr DoubleMantissa<Real> epsilon(epsilon_single * epsilon_single /
                                         Real{2});

  return epsilon;
}

template <typename Real, typename>
constexpr DoubleMantissa<Real> Infinity() {
  constexpr DoubleMantissa<Real> infinity =
      DoubleMantissa<Real>(std::numeric_limits<Real>::infinity());
  return infinity;
}

template <typename Real, typename>
constexpr DoubleMantissa<Real> QuietNan() {
  constexpr DoubleMantissa<Real> quiet_nan =
      DoubleMantissa<Real>(std::numeric_limits<Real>::quiet_NaN());
  return quiet_nan;
}

template <typename Real, typename>
constexpr DoubleMantissa<Real> SignalingNan() {
  constexpr DoubleMantissa<Real> signaling_nan =
      DoubleMantissa<Real>(std::numeric_limits<Real>::signaling_NaN());
  return signaling_nan;
}

template <typename Real, typename>
DoubleMantissa<Real> ComputeLogOf2() {
  // Calculate log(2) via Eq. (30) from:
  //
  //   X. Gourdon and P. Sebah, "The logarithmic constant: log2", Jan. 2004.
  //
  // That is,
  //
  //   log(2) = (3/4) \sum_{j >= 0} (-1)^j \frac{(j!)^2}{2^j (2 j + 1)!}.
  //
  // We follow MPFR -- see Subsection 5.3, "The log2 constant", of
  //
  //   The MPFR Team, "The MPFR Library: Algorithms and Proofs",
  //   URL: https://www.mpfr.org/algorithms.pdf
  //
  // by truncating after the first N = floor(w / 3) + 1 iterations, where w
  // is the number of binary significant digits -- the "working precision".
  //
  // However, we do not have access to arbitrary-precision integer types, nor
  // would we want to make use of dynamic memory allocation, so we cannot
  // exactly store the (integer) numerator and denominator. And, in single-
  // precision, it is easy to accidentally overflow in the later iterates.
  constexpr int needed_digits =
      std::numeric_limits<DoubleMantissa<Real>>::digits;
  constexpr int num_terms = (needed_digits / 3) + 1;

  // Accumulate the terms of the series.
  //
  // The j'th term in the series is related to the (j-1)'th term in the series
  // via:
  //
  //   t_j = (j^2 / (2 * 2 j * (2j + 1))) t_{j - 1}.
  //
  // In order to ensure that the denominators of our summed terms from the
  // series are consistent,
  //
  //       (0!)^2             (1!)^2
  //  ----------------- + ----------------
  //   2^0 (2 * 0 + 1)!   2^1 (2 * 1 + 1)!
  DoubleMantissa<Real> iterate = Real{1};
  DoubleMantissa<Real> term = Real{1};
  for (int j = 1; j < num_terms; ++j) {
    // Accumulate the term.
    // Typically both real numbers are exactly represented.
    term *= Real(j * j);
    term /= Real(2 * (2 * j) * (2 * j + 1));
    term *= -1;

    iterate += term;
  }
  iterate *= 3;
  iterate /= 4;

  return iterate;
}

template <typename Real, typename>
DoubleMantissa<Real> LogOf2() {
  static const DoubleMantissa<Real> log_of_2 = ComputeLogOf2<Real>();
  return log_of_2;
}

template <typename Real, typename>
DoubleMantissa<Real> LogOf10() {
  // TODO(Jack Poulson): Switch to a fully-accurate representation.
  static const DoubleMantissa<Real> log_of_10 = Log(DoubleMantissa<Real>(10));
  return log_of_10;
}

template <typename Real, typename>
DoubleMantissa<Real> ComputePi() {
  // We use the algorithm of
  //
  //   A. Schonhage, A. F. W. Grotefeld, and E. Vetter,
  //   "Fast Algorithms: A Multitape Turing Machine Implementation",
  //   BI Wissenschaftverlag, 1994.
  //
  // as described in
  //
  //   The MPFR Team, "The MPFR Library: Algorithms and Proofs".
  //
  constexpr int digits = std::numeric_limits<DoubleMantissa<Real>>::digits;

  DoubleMantissa<Real> a_squared = 1;
  DoubleMantissa<Real> b_squared = Real{1} / Real{2};
  DoubleMantissa<Real> denominator = Real{1} / Real{4};

  DoubleMantissa<Real> a = 1;
  DoubleMantissa<Real> b, s, diff;
  Real tolerance = Real{2} * std::pow(Real{2}, -digits);
  for (unsigned iter = 0;; ++iter, tolerance *= Real{2}) {
    s = (a_squared + b_squared) / Real{4};
    b = SquareRoot(b_squared);
    a = (a + b) / Real{2};
    a_squared = Square(a);
    b_squared = Real{2} * (a_squared - s);
    diff = a_squared - b_squared;
    denominator -= LoadExponent(diff, iter);
    if (std::abs(diff.Upper()) <= tolerance) {
      break;
    }
  }

  const DoubleMantissa<Real> pi = b_squared / denominator;

  return pi;
}

template <typename Real, typename>
DoubleMantissa<Real> Pi() {
  static const DoubleMantissa<Real> pi = ComputePi<Real>();
  return pi;
}

template <typename Real, typename>
DoubleMantissa<Real> EulerNumber() {
  static const DoubleMantissa<Real> e = std::exp(DoubleMantissa<Real>(1));
  return e;
}

}  // namespace double_mantissa

}  // namespace mantis

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_CLASS_IMPL_H_
