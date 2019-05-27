/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_IMPL_H_

#include <iomanip>
#include <string>

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
bool DoubleMantissa<Real>::operator<(const DoubleMantissa<Real>& value) const {
  return Upper() < value.Upper() ||
         (Upper() == value.Upper() && Lower() < value.Lower());
}

template <typename Real>
bool DoubleMantissa<Real>::operator<=(const DoubleMantissa<Real>& value) const {
  return Upper() < value.Upper() ||
         (Upper() == value.Upper() && Lower() <= value.Lower());
}

template <typename Real>
bool DoubleMantissa<Real>::operator>(const DoubleMantissa<Real>& value) const {
  return Upper() > value.Upper() ||
         (Upper() == value.Upper() && Lower() > value.Lower());
}

template <typename Real>
bool DoubleMantissa<Real>::operator>=(const DoubleMantissa<Real>& value) const {
  return Upper() > value.Upper() ||
         (Upper() == value.Upper() && Lower() >= value.Lower());
}

template <typename Real>
DoubleMantissa<Real>::operator int() const {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return int(floored_value.Upper()) + int(floored_value.Lower());
}

template <typename Real>
DoubleMantissa<Real>::operator long int() const {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return static_cast<long int>(floored_value.Upper()) +
         static_cast<long int>(floored_value.Lower());
}

template <typename Real>
DoubleMantissa<Real>::operator long long int() const {
  const DoubleMantissa<Real> floored_value = Floor(*this);
  return static_cast<long long int>(floored_value.Upper()) +
         static_cast<long long int>(floored_value.Lower());
}

template <typename Real>
DoubleMantissa<Real>::operator float() const {
  return Upper();
}

template <typename Real>
DoubleMantissa<Real>::operator double() const {
  return Upper();
}

template <typename Real>
DoubleMantissa<Real>::operator long double() const {
  return Upper();
}

template <typename Real>
mantis::DecimalScientificNotation
DoubleMantissa<Real>::DecimalScientificNotation(int num_digits) const {
  DoubleMantissa<Real> value = *this;

  mantis::DecimalScientificNotation rep;

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
DoubleMantissa<Real>& DoubleMantissa<Real>::FromDecimalScientificNotation(
    const mantis::DecimalScientificNotation& rep) {
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
DoubleMantissa<Real> Log10(const DoubleMantissa<Real>& value) {
  return Log(value) / double_mantissa::LogOf10<Real>();
}

template <typename Real>
DoubleMantissa<Real> Abs(const DoubleMantissa<Real>& value) {
  return value.Upper() < Real(0) ? -value : value;
}

template <typename Real>
DoubleMantissa<Real> SmallArgSin(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  // The Taylor series is of the form: x - (1/3!) x^3 + (1/5!) x^5 - ...,
  // we the successive powers of 'x' will be multiplied by -x^2.
  const DoubleMantissa<Real> power_ratio = -Square(value);

  // Initialize x_power := x and the iterate as the first term in the series.
  DoubleMantissa<Real> x_power = value;
  DoubleMantissa<Real> iterate(value);

  const Real threshold = Real(0.5) * std::abs(value.Upper()) *
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
DoubleMantissa<Real> SmallArgCos(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  // The Taylor series is of the form: 1 - (1/2!) x^2 + (1/4!) x^4 - ...,
  // we the successive powers of 'x' will be multiplied by -x^2.
  const DoubleMantissa<Real> power_ratio = -Square(value);

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
void DecomposeSinCosArgument(const DoubleMantissa<Real>& value,
                             int* num_half_pi_int, int* num_sixteenth_pi_int,
                             DoubleMantissa<Real>* sixteenth_pi_remainder) {
  static const DoubleMantissa<Real> pi = double_mantissa::Pi<Real>();
  static const DoubleMantissa<Real> two_pi = Real(2) * pi;
  const DoubleMantissa<Real> num_two_pi = Round(value / two_pi);
  const DoubleMantissa<Real> two_pi_remainder = value - two_pi * num_two_pi;

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
DoubleMantissa<Real> Sin(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(value, &num_half_pi_int, &num_sixteenth_pi_int,
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

  DoubleMantissa<Real> sin_value;
  if (num_half_pi_int == 0) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      sin_value = sin_a * cos_b + cos_a * sin_b;
    } else {
      // sin(-a + b) = sin(-a) cos(b) + cos(-a) sin(b)
      //             = -sin(a) cos(b) + cos(a) sin(b).
      sin_value = -sin_a * cos_b + cos_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + (pi / 2)) = cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      sin_value = cos_a * cos_b - sin_a * sin_b;
    } else {
      // sin(-a + b + (pi / 2)) = cos(-a + b)
      //     = cos(a) cos(b) + sin(a) sin(b).
      sin_value = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b - (pi / 2)) = -cos(a + b)
      //     = -cos(a) cos(b) + sin(a) sin(b).
      sin_value = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // sin(-a + b - (pi / 2)) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      sin_value = -cos_a * cos_b - sin_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + pi) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      sin_value = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // sin(-a + b + pi) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      sin_value = sin_a * cos_b - cos_a * sin_b;
    }
  }

  return sin_value;
}

template <typename Real>
DoubleMantissa<Real> Cos(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(value, &num_half_pi_int, &num_sixteenth_pi_int,
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

  DoubleMantissa<Real> cos_value;
  if (num_half_pi_int == 0) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      cos_value = cos_a * cos_b - sin_a * sin_b;
    } else {
      // cos(-a + b) = cos(a) cos(b) + sin(a) sin(b).
      cos_value = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b + (pi / 2)) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      cos_value = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // cos(-a + b + (pi / 2)) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      cos_value = sin_a * cos_b - cos_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b - (pi / 2)) = sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      cos_value = sin_a * cos_b + cos_a * sin_b;
    } else {
      // cos(-a + b - (pi / 2)) = sin(-a + b) = -sin(a) cos(b) + cos(a) sin(b).
      cos_value = -sin_a * cos_b + cos_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // cos(a + b + pi) = -cos(a + b) = -cos(a) cos(b) + sin(a) sin(b).
      cos_value = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // cos(-a + b + pi) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      cos_value = -cos_a * cos_b - sin_a * sin_b;
    }
  }

  return cos_value;
}

template <typename Real>
void SinCos(const DoubleMantissa<Real>& value, DoubleMantissa<Real>* s,
            DoubleMantissa<Real>* c) {
  if (value.Upper() == Real(0)) {
    *s = DoubleMantissa<Real>();
    *c = DoubleMantissa<Real>(Real(1));
    return;
  }

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(value, &num_half_pi_int, &num_sixteenth_pi_int,
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
      *s = sin_b;
      // cos(a + b) = cos(b).
      *c = cos_b;
    } else if (num_half_pi_int == 1) {
      // sin(a + b + (pi / 2)) = sin(b + (pi / 2)) = cos(b).
      *s = cos_b;
      // cos(a + b + (pi / 2)) = cos(b + (pi / 2)) = -sin(b).
      *c = -sin_b;
    } else if (num_half_pi_int == -1) {
      // sin(a + b - (pi / 2)) = sin(b - (pi / 2)) = -cos(b).
      *s = -cos_b;
      // cos(a + b - (pi / 2)) = cos(b - (pi / 2)) = sin(b).
      *c = sin_b;
    } else {
      // sin(a + b + pi) = sin(b + pi) = -sin(b).
      *s = -sin_b;
      // cos(a + b + pi) = cos(b + pi) = -cos(b).
      *c = -cos_b;
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
      *s = sin_a * cos_b + cos_a * sin_b;
      // cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      *c = cos_a * cos_b - sin_a * sin_b;
    } else {
      // sin(-a + b) = sin(-a) cos(b) + cos(-a) sin(b)
      //             = -sin(a) cos(b) + cos(a) sin(b).
      *s = -sin_a * cos_b + cos_a * sin_b;
      // cos(-a + b) = cos(a) cos(b) + sin(a) sin(b).
      *c = cos_a * cos_b + sin_a * sin_b;
    }
  } else if (num_half_pi_int == 1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + (pi / 2)) = cos(a + b) = cos(a) cos(b) - sin(a) sin(b).
      *s = cos_a * cos_b - sin_a * sin_b;
      // cos(a + b + (pi / 2)) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      *c = -sin_a * cos_b - cos_a * sin_b;
    } else {
      // sin(-a + b + (pi / 2)) = cos(-a + b)
      //     = cos(a) cos(b) + sin(a) sin(b).
      *s = cos_a * cos_b + sin_a * sin_b;
      // cos(-a + b + (pi / 2)) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      *c = sin_a * cos_b - cos_a * sin_b;
    }
  } else if (num_half_pi_int == -1) {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b - (pi / 2)) = -cos(a + b)
      //     = -cos(a) cos(b) + sin(a) sin(b).
      *s = -cos_a * cos_b + sin_a * sin_b;
      // cos(a + b - (pi / 2)) = sin(a + b) = sin(a) cos(b) + cos(a) sin(b).
      *c = sin_a * cos_b + cos_a * sin_b;
    } else {
      // sin(-a + b - (pi / 2)) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      *s = -cos_a * cos_b - sin_a * sin_b;
      // cos(-a + b - (pi / 2)) = sin(-a + b) = -sin(a) cos(b) + cos(a) sin(b).
      *c = -sin_a * cos_b + cos_a * sin_b;
    }
  } else {
    if (num_sixteenth_pi_int > 0) {
      // sin(a + b + pi) = -sin(a + b) = -sin(a) cos(b) - cos(a) sin(b).
      *s = -sin_a * cos_b - cos_a * sin_b;
      // cos(a + b + pi) = -cos(a + b) = -cos(a) cos(b) + sin(a) sin(b).
      *c = -cos_a * cos_b + sin_a * sin_b;
    } else {
      // sin(-a + b + pi) = -sin(-a + b) = sin(a) cos(b) - cos(a) sin(b).
      *s = sin_a * cos_b - cos_a * sin_b;
      // cos(-a + b + pi) = -cos(-a + b) = -cos(a) cos(b) - sin(a) sin(b).
      *c = -cos_a * cos_b - sin_a * sin_b;
    }
  }
}

template <typename Real>
DoubleMantissa<Real> Tan(const DoubleMantissa<Real>& value) {
  DoubleMantissa<Real> s, c;
  SinCos(value, &s, &c);
  return s / c;
}

template <typename Real>
DoubleMantissa<Real> ArcTan(const DoubleMantissa<Real>& tan_theta) {
  return ArcTan2(tan_theta, DoubleMantissa<Real>(1));
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
DoubleMantissa<Real> HyperbolicSin(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(0));
  }

  // We use the same switching point as the QD library of Hida et al.
  const Real kDirectFormulaThresh = Real(0.05);
  if (std::abs(value.Upper()) > kDirectFormulaThresh) {
    const DoubleMantissa<Real> exp_value = Exp(value);
    return MultiplyByPowerOfTwo(exp_value - Inverse(exp_value), Real(0.5));
  }

  // We fall back to a truncation of the Taylor series:
  //
  //   sinh(x) = x + (1 / 3!) x^3 + (1 / 5!) x^5 + ...
  //           = \sum_{n = 0}^{\infty} x^{2 n + 1} / (2 n + 1)!.
  //
  const DoubleMantissa<Real> epsilon =
      std::numeric_limits<DoubleMantissa<Real>>::epsilon();
  const Real term_threshold = std::abs(value.Upper()) * epsilon.Upper();

  // The powers of x increase by x^2 in each term.
  const DoubleMantissa<Real> power_ratio = Square(value);

  // Initialize the partial sum as the first term, x.
  DoubleMantissa<Real> term = value;
  DoubleMantissa<Real> iterate = value;

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
DoubleMantissa<Real> HyperbolicCos(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(1));
  }

  // Use the defining formula.
  const DoubleMantissa<Real> exp_value = Exp(value);
  return MultiplyByPowerOfTwo(exp_value + Inverse(exp_value), Real(0.5));
}

template <typename Real>
DoubleMantissa<Real> HyperbolicTan(const DoubleMantissa<Real>& value) {
  if (value.Upper() == Real(0)) {
    return DoubleMantissa<Real>(Real(0));
  }

  // We use the same switching point as the QD library of Hida et al.
  const Real kDirectFormulaThresh = Real(0.05);
  if (Abs(value) > kDirectFormulaThresh) {
    const DoubleMantissa<Real> exp_value = Exp(value);
    const DoubleMantissa<Real> inv_exp_value = Inverse(exp_value);
    return (exp_value - inv_exp_value) / (exp_value + inv_exp_value);
  }

  // Fall back to a careful calculation of hyperbolic sine.
  const DoubleMantissa<Real> sinh_value = HyperbolicSin(value);

  // Convert the calculation of sinh into cosh using the identity:
  //
  //   cosh^2(x) - sinh^2(x) = 1.
  //
  const DoubleMantissa<Real> cosh_value =
      SquareRoot(Real(1) + Square(sinh_value));

  return sinh_value / cosh_value;
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
  if (Abs(tanh_x.Upper()) >= Real(1)) {
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

template <typename Real>
DoubleMantissa<Real> Floor(const DoubleMantissa<Real>& value) {
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
DoubleMantissa<Real> Round(const DoubleMantissa<Real>& value) {
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
  constexpr int max_digits10 =
      std::numeric_limits<mantis::DoubleMantissa<Real>>::max_digits10;
  const mantis::DecimalScientificNotation rep =
      value.DecimalScientificNotation(max_digits10);

  std::string s;
  if (!rep.positive) {
    s += '-';
  }
  if (rep.digits.size() == 3 && rep.digits[0] == 'i') {
    s += "inf";
  } else if (rep.digits.size() == 3 && rep.digits[0] == 'n') {
    s += "nan";
  } else {
    s += std::to_string(unsigned(rep.digits[0]));
    s += '.';
    for (unsigned digit = 1; digit < rep.digits.size(); ++digit) {
      s += std::to_string(unsigned(rep.digits[digit]));
    }
    if (rep.exponent != 0) {
      s += 'e';
      s += std::to_string(rep.exponent);
    }
  }

  out << s;

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

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_IMPL_H_
