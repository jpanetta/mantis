/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_IMPL_H_

#include "mantis/double_mantissa.hpp"

#include "mantis/util.hpp"

namespace mantis {

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa() : values_{Real{0}, Real{0}} {}

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa(const Real& upper)
    : values_{upper, Real{0}} {}

template <typename Real>
constexpr DoubleMantissa<Real>::DoubleMantissa(const Real& upper,
                                               const Real& lower)
    : values_{upper, lower} {}

template <typename Real>
DoubleMantissa<Real>::DoubleMantissa(const DoubleMantissa<Real>& value)
    : values_{value.Upper(), value.Lower()} {}

template <typename Real>
Real& DoubleMantissa<Real>::Upper() {
  return values_[0];
}

template <typename Real>
const Real& DoubleMantissa<Real>::Upper() const {
  return values_[0];
}

template <typename Real>
Real& DoubleMantissa<Real>::Lower() {
  return values_[1];
}

template <typename Real>
const Real& DoubleMantissa<Real>::Lower() const {
  return values_[1];
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::Reduce() {
  Upper() = TwoSum(Upper(), Lower(), &Lower());
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(const Real& value) {
  Upper() = value;
  Lower() = 0;
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator=(
    const DoubleMantissa<Real>& value) {
  Upper() = value.Upper();
  Lower() = value.Lower();
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(const Real& value) {
  Real error;
  Real upper_sum = TwoSum(Upper(), value, &error);
  error += Lower();
  Upper() = QuickTwoSum(upper_sum, error, &Lower());
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator+=(
    const DoubleMantissa<Real>& value) {
#ifdef MANTIS_IEEE_SUM
  // This algorithm is attributed by Hida et al. to Briggs and Kahan.
  Real upper_error;
  Real upper_sum = TwoSum(Upper(), value.Upper(), &upper_error);

  Real lower_error;
  Real lower_sum = TwoSum(Lower(), value.Lower(), &lower_error);
  upper_error += lower_sum;

  upper_sum = QuickTwoSum(upper_sum, upper_error, &upper_error);
  upper_error += lower_error;

  Upper() = QuickTwoSum(upper_sum, upper_error, &Lower());
#else
  // In QD, this is referred to as a 'sloppy' add, as it only obey's a
  // Cray-style error bound.
  Real error;
  Real upper_sum = TwoSum(Upper(), value.Upper(), &error);
  error += Lower();
  error += value.Lower();
  Upper() = QuickTwoSum(upper_sum, error, &Lower());
#endif  // ifdef MANTIS_IEEE_SUM
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(const Real& value) {
  Real error;
  Real upper_diff = TwoDiff(Upper(), value, &error);
  error += Lower();
  Upper() = QuickTwoSum(upper_diff, error, &Lower());
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator-=(
    const DoubleMantissa<Real>& value) {
#ifdef MANTIS_IEEE_SUM
  Real upper_error;
  Real upper_diff = TwoDiff(Upper(), value.Upper(), &upper_error);

  Real lower_error;
  Real lower_diff = TwoDiff(Lower(), value.Lower(), &lower_error);

  upper_error += lower_diff;
  new_upper = QuickTwoSum(upper_diff, upper_error, &upper_error);

  upper_error += lower_error;
  Upper() = QuickTwoSum(upper_diff, upper_error, &Lower());
#else
  Real error;
  Real upper_diff = TwoDiff(Upper(), value.Upper(), &error);
  error += Lower();
  error -= value.Lower();
  Upper() = QuickTwoSum(upper_diff, error, &Lower());
#endif  // ifdef MANTIS_IEEE_SUM
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(const Real& value) {
  Real error;
  const Real product = TwoProd(Upper(), value, &error);
  error += Lower() * value;
  Upper() = QuickTwoSum(product, error, &Lower());
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator*=(
    const DoubleMantissa<Real>& value) {
  Real error;
  const Real product = TwoProd(Upper(), value.Upper(), &error);
  error += value.Lower() * Upper();
  error += value.Upper() * Lower();
  Upper() = QuickTwoSum(product, error, &Lower());
  return *this;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(const Real& value) {
  // See the QD library by Hida et al. for a similar implementation.
  const Real approx_quotient = Upper() / value;

  // Compute the remainder, remander := *this - approx_quotient * value.
  Real product_error;
  const Real product = TwoProd(approx_quotient, value, &product_error);
  Real remainder_error;
  const Real remainder = TwoDiff(Upper(), product, &remainder_error);
  remainder_error += Lower();
  remainder_error -= product_error;

  // Perform iterative refinement.
  const Real update = (remainder + remainder_error) / value;
  Upper() = QuickTwoSum(approx_quotient, update, &Lower());

  return *this;
}

template <typename Real>
bool DoubleMantissa<Real>::operator==(const DoubleMantissa<Real>& value) const {
  return Upper() == value.Upper() && Lower() == value.Lower();
}

template <typename Real>
bool DoubleMantissa<Real>::operator!=(const DoubleMantissa<Real>& value) const {
  return Upper() != value.Upper() || Lower() != value.Lower();
}

template <typename Real>
DoubleMantissa<Real> DoubleMantissa<Real>::FastDivide(
    const DoubleMantissa<Real>& x, const DoubleMantissa<Real>& y) {
  // See the QD library by Hida et al. for a similar implementation.
  const Real approx_quotient = x.Upper() / y.Upper();

  // Compute the remainder, remander := *this - approx_quotient * value.
  const DoubleMantissa<Real> product = y * approx_quotient;
  Real remainder_error;
  const Real remainder = TwoDiff(x.Upper(), product.Upper(), &remainder_error);
  remainder_error += x.Lower();
  remainder_error -= product.Lower();

  // Perform iterative refinement.
  const Real update = (remainder + remainder_error) / y.Upper();
  DoubleMantissa<Real> result;
  result.Upper() = QuickTwoSum(approx_quotient, update, &result.Lower());

  return result;
}

template <typename Real>
DoubleMantissa<Real> DoubleMantissa<Real>::Divide(
    const DoubleMantissa<Real>& x, const DoubleMantissa<Real>& y) {
  // See the QD library by Hida et al. for a similar implementation.
  Real approx_quotient = x.Upper() / y.Upper();

  // Compute the original remainder.
  DoubleMantissa<Real> r = x - approx_quotient * y;

  // Perform a step of iterative refinement.
  Real update0 = r.Upper() / y.Upper();
  r -= update0 * y;

  // Perform another step of iterative refinement.
  const Real update1 = r.Upper() / y.Upper();

  // Combine the original approximation and the two updates into the result,
  // which we store in the same place as the original remainder. We also
  // overwrite the first update with the original sum error after its
  // incorporation.
  approx_quotient = QuickTwoSum(approx_quotient, update0, &update0);
  r = DoubleMantissa<Real>(approx_quotient, update0);
  r += update1;

  return r;
}

template <typename Real>
DoubleMantissa<Real>& DoubleMantissa<Real>::operator/=(
    const DoubleMantissa<Real>& value) {
  *this = Divide(*this, value);
  return *this;
}

template <typename Real>
DoubleMantissa<Real> Square(const Real& value) {
  Real error;
  const Real upper = TwoSquare(value, &error);
  return DoubleMantissa<Real>(upper, error);
}

template <typename Real>
DoubleMantissa<Real> Square(const DoubleMantissa<Real>& value) {
  Real error;
  const Real product = TwoSquare(value.Upper(), &error);
  error += Real{2} * value.Lower() * value.Upper();

  Real new_error;
  const Real new_upper = QuickTwoSum(product, error, &new_error);
  return DoubleMantissa<Real>(new_upper, new_error);
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
  // contain a typo suggesting 'x' approximates sqrt(a) -- the square-root is
  // computed as:
  //
  //   a * x + [a - (a * x)^2] * (x / 2).
  //
  if (value.Upper() == Real{0} && value.Lower() == Real{0}) {
    return value;
  }

  if (value.Upper() < Real{0}) {
    return double_mantissa::QuietNan<Real>();
  }

  const Real inv_sqrt = Real{1} / std::sqrt(value.Upper());
  const Real left_term = value.Upper() * inv_sqrt;
  const Real right_term =
      (value - Square(left_term)).Upper() * (inv_sqrt / Real{2});
  return DoubleMantissa<Real>(left_term, right_term).Reduce();
}

template <typename Real>
DoubleMantissa<Real> MultiplyByPowerOfTwo(const DoubleMantissa<Real>& value,
                                          const Real& power_of_two) {
  return DoubleMantissa<Real>(value.Upper() * power_of_two,
                              value.Lower() * power_of_two);
}

template <typename Real>
mantis::DoubleMantissa<Real> LoadExponent(
    const mantis::DoubleMantissa<Real>& value, int exp) {
  return mantis::DoubleMantissa<Real>(std::ldexp(value.Upper(), exp),
                                      std::ldexp(value.Lower(), exp));
}

}  // namespace mantis

template <typename Real>
mantis::DoubleMantissa<Real> operator-(
    const mantis::DoubleMantissa<Real>& value) {
  return mantis::DoubleMantissa<Real>(-value.Upper(), -value.Lower());
}

template <typename Real>
mantis::DoubleMantissa<Real> operator+(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(y);
  z += x;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator+(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y) {
  mantis::DoubleMantissa<Real> z(x);
  z += y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator+(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(x);
  z += y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator-(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(-y);
  z += x;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator-(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y) {
  mantis::DoubleMantissa<Real> z(x);
  z -= y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator-(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(x);
  z -= y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator*(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(y);
  z *= x;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator*(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y) {
  mantis::DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator*(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  mantis::DoubleMantissa<Real> z(x);
  z *= y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator/(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  return mantis::DoubleMantissa<Real>::Divide(mantis::DoubleMantissa<Real>(x),
                                              y);
}

template <typename Real>
mantis::DoubleMantissa<Real> operator/(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y) {
  mantis::DoubleMantissa<Real> z(x);
  z /= y;
  return z;
}

template <typename Real>
mantis::DoubleMantissa<Real> operator/(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y) {
  return mantis::DoubleMantissa<Real>::Divide(x, y);
}

template <typename Real>
std::ostream& operator<<(std::ostream& out,
                         const mantis::DoubleMantissa<Real>& value) {
  out << value.Upper() << " + " << value.Lower();
  return out;
}

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
mantis::DoubleMantissa<Real> ldexp(const mantis::DoubleMantissa<Real>& value,
                                   int exp) {
  return mantis::LoadExponent(value, exp);
}

template <typename Real>
mantis::DoubleMantissa<Real> sqrt(const mantis::DoubleMantissa<Real>& value) {
  return mantis::SquareRoot(value);
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_IMPL_H_
