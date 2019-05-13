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
DoubleMantissa<Real> LoadExponent(const DoubleMantissa<Real>& value, int exp) {
  return DoubleMantissa<Real>(std::ldexp(value.Upper(), exp),
                              std::ldexp(value.Lower(), exp));
}

template <typename Real>
DoubleMantissa<Real> Exp(const DoubleMantissa<Real>& value) {
  // We roughly reproduce the range provided within the QD library of Hida et
  // al., where the maximum double-precision exponent is set to 709 despite
  // std::log(numeric_limits<double>::max()) = 709.783.
  static constexpr Real log_max = LogMax<Real>();
  static constexpr Real exp_max = log_max - Real{0.783};

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

  // TODO(Jack Poulson): If the input is equal to one, return 'e'.

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

  for (int j = 4; j < num_terms; ++j) {
    // (r_power)_j = r^j.
    r_power *= r;

    // coefficient_j = 1 / j!.
    coefficient /= Real(j);

    // term_j = (1 / j!) r^j.
    term = coefficient * r_power;

    // iterate_j := r + (1/2!) r^2 + ... + (1/j!) r^j.
    iterate += term;

    // TODO(Jack Poulson): Break if the term is small enough.
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

  DoubleMantissa<Real> x = std::log(value.Upper());

  const int num_iter = 1;
  for (int j = 0; j < num_iter; ++j) {
    // TODO(Jack Poulson): Analyze whether this order of operations is ideal.
    // Hopefully we can avoid catastrophic cancellation.
    x += value * Exp(-x);
    x -= Real{1};
  }

  return x;
}

namespace double_mantissa {

template <typename Real>
DoubleMantissa<Real> LogOf2() {
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
  constexpr int needed_digits =
      std::numeric_limits<DoubleMantissa<Real>>::digits;
  constexpr int num_terms = (needed_digits / 3) + 1;

  // We precompute the numerator and denominator for the first five terms.
  // The initial denominator is: prod_{j=1}^5 2 * (2 * j) * (2 * j + 1).
  // The initial numerator is computed as in the iteration below, starting
  // at j=1 with the state numerator=1.
  constexpr unsigned int numerator5 = 1180509120uL;
  constexpr unsigned int denominator5 = 1277337600uL;
  constexpr unsigned int factorial5 = 120uL;  // 5! = 120

  // Accumulate the remaining terms of the numerator and denominator.
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
  DoubleMantissa<Real> numerator = numerator5;
  DoubleMantissa<Real> denominator = denominator5;
  DoubleMantissa<Real> factorial = factorial5;
  int sign = -1;
  for (int j = 6; j < num_terms; ++j) {
    // Set sign := (-1)^j.
    sign = -sign;

    // Set factorial := j!.
    factorial *= j;

    // Set temp equal to the ratio of the j'th denominator to the (j-1)'th
    // denominator, 2 * (2 * j) * (2 * j + 1), which holds for j >= 1.
    DoubleMantissa<Real> temp = 2 * (2 * j) * (2 * j + 1);

    // Rescale the fraction to account for the new denominator.
    numerator *= temp;
    denominator *= temp;

    // Compute temp := (-1)^j (j!)^2
    temp = factorial * factorial;
    temp *= sign;

    // Due to the multiplicative nesting property of the denominators in the
    // series, we may directly accumulate this term into the numerator.
    numerator += temp;
  }
  numerator *= 3;
  denominator *= 4;

  static const DoubleMantissa<Real> log_of_two = numerator / denominator;
  return log_of_two;
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
  out << std::setprecision(max_digits10) << value.Upper() << " + "
      << value.Lower();
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
mantis::DoubleMantissa<Real> exp(const mantis::DoubleMantissa<Real>& value) {
  return mantis::Exp(value);
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
mantis::DoubleMantissa<Real> sqrt(const mantis::DoubleMantissa<Real>& value) {
  return mantis::SquareRoot(value);
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_IMPL_H_
