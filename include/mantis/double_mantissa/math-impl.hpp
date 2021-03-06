/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_MATH_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_MATH_IMPL_H_

#include <random>

#include "mantis/util.hpp"

#include "mantis/double_mantissa.hpp"

namespace mantis {

template <typename Real>
Real Hypot(const Real& x, const Real& y) {
  const Real x_abs = Abs(x);
  const Real y_abs = Abs(y);

  const Real& a = y_abs > x_abs ? y : x;
  const Real& b = y_abs > x_abs ? x : y;
  const Real& a_abs = y_abs > x_abs ? y_abs : x_abs;

  if (a == 0) {
    return Real();
  }

  const Real t = b / a;
  return a_abs * SquareRoot(1 + Square(t));
}

template <typename Real>
Real LogMax() {
  static const Real log_max = std::log(std::numeric_limits<Real>::max());
  return log_max;
}

template <typename Real, typename>
Real LogOf2() {
  static const Real log_of_2 = std::log(Real(2));
  return log_of_2;
}

template <typename Real>
Real ComputeLogOf2() {
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
  constexpr int needed_digits = std::numeric_limits<Real>::digits;
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
  Real iterate = 1;
  Real term = 1;
  for (int j = 1; j < num_terms; ++j) {
    // Accumulate the term.
    // Typically both real numbers are exactly represented.
    term *= j * j;
    term /= 2 * (2 * j) * (2 * j + 1);
    term *= -1;

    iterate += term;
  }
  iterate *= 3;
  iterate /= 4;

  return iterate;
}

template <typename Real>
Real LogOf2() {
  static const Real log_of_2 = ComputeLogOf2<Real>();
  return log_of_2;
}

template <typename Real>
Real LogOf10() {
  static const Real log_of_10 = std::log(Real(10));
  return log_of_10;
}

template <typename Real, typename>
Real ComputePi() {
  typedef typename Real::Base Base;

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
  constexpr int digits = std::numeric_limits<Real>::digits;

  Real a_squared = 1;
  Real b_squared = Base{1} / 2;
  Real denominator = Base{1} / 4;

  Real a = 1;
  Real b, s, diff;
  Base tolerance = 2 * std::pow(Base{2}, -digits);
  for (unsigned iter = 0;; ++iter, tolerance *= 2) {
    s = (a_squared + b_squared) / 4;
    b = SquareRoot(b_squared);
    a = (a + b) / 2;
    a_squared = Square(a);
    b_squared = 2 * (a_squared - s);
    diff = a_squared - b_squared;
    denominator -= LoadExponent(diff, iter);
    if (std::abs(diff.Upper()) <= tolerance) {
      break;
    }
  }

  const Real pi = b_squared / denominator;
  return pi;
}

template <typename Real, typename>
Real Pi() {
  static const Real pi = std::acos(Real(-1));
  return pi;
}

template <typename Real>
Real Pi() {
  static const Real pi = ComputePi<Real>();
  return pi;
}

template <typename Real>
Real EulerNumber() {
  static const Real e = std::exp(Real(1));
  return e;
}

template <typename Real, typename>
constexpr Real Epsilon() {
  return std::numeric_limits<Real>::epsilon();
}

template <typename Real>
constexpr Real Epsilon() {
  typedef typename Real::Base Base;
  constexpr Base epsilon_single = std::numeric_limits<Base>::epsilon();

  // The 'numeric_limits' definition of epsilon_single is 2^{1 - p} if the
  // binary precision is p digits. Thus, if the precision is doubled, epsilon
  // becomes 2^{1 - 2 p} = (2^{1 - p})^2 / 2 = epsilon_single^2 / 2.
  //
  // This approach is preferable because it allows us to compute the new
  // precision as a constexpr (avoiding pow).
  constexpr DoubleMantissa<Base> epsilon(epsilon_single * epsilon_single /
                                         Base{2});

  return epsilon;
}

template <typename Real, typename>
constexpr Real Infinity() {
  return std::numeric_limits<Real>::infinity();
}

template <typename Real>
constexpr Real Infinity() {
  return std::numeric_limits<typename Real::Base>::infinity();
}

template <typename Real, typename>
constexpr Real QuietNan() {
  return std::numeric_limits<Real>::quiet_NaN();
}

template <typename Real>
constexpr Real QuietNan() {
  return std::numeric_limits<typename Real::Base>::quiet_NaN();
}

template <typename Real, typename>
constexpr Real SignalingNan() {
  return std::numeric_limits<Real>::signaling_NaN();
}

template <typename Real>
constexpr Real SignalingNan() {
  return std::numeric_limits<typename Real::Base>::signaling_NaN();
}

template <typename Real>
DoubleMantissa<Real> Inverse(const DoubleMantissa<Real>& value) {
  return Real(1) / value;
}

inline float Square(const float& value) { return value * value; }

inline double Square(const double& value) { return value * value; }

inline long double Square(const long double& value) { return value * value; }

template <typename Real>
DoubleMantissa<Real> Square(const DoubleMantissa<Real>& value) {
  DoubleMantissa<Real> product = TwoSquare(value.Upper());
  product.Lower() += Real{2} * value.Lower() * value.Upper();
  return QuickTwoSum(product);
}

template <typename Real, typename>
DoubleMantissa<Real> SquareIntoDouble(const Real& value) {
  return TwoSquare(value);
}

inline float SquareRoot(const float& value) { return std::sqrt(value); }

inline double SquareRoot(const double& value) { return std::sqrt(value); }

inline long double SquareRoot(const long double& value) {
  return std::sqrt(value);
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
    return QuietNan<DoubleMantissa<Real>>();
  }

  DoubleMantissa<Real> iterate;
  {
    const Real inv_sqrt = Real{1} / std::sqrt(value.Upper());
    const Real left_term = value.Upper() * inv_sqrt;
    const Real right_term =
        (value - SquareIntoDouble(left_term)).Upper() * (inv_sqrt / Real{2});
    iterate = DoubleMantissa<Real>(left_term, right_term).Reduce();
  }

  const bool two_iterations = true;
  if (two_iterations) {
    iterate = DoubleMantissa<Real>(1) / iterate;
    const Real left_term = value.Upper() * iterate.Upper();
    const Real right_term = (value - SquareIntoDouble(left_term)).Upper() *
                            (iterate.Upper() / Real{2});
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
    return Infinity<DoubleMantissa<Real>>();
  }
  if (value.Upper() == Real{0}) {
    return DoubleMantissa<Real>(Real{1});
  }

  // TODO(Jack Poulson): Consider returning a fully-precise result for
  // e = exp(1). At the moment, we compute this constant by calling this
  // routine.

  static const DoubleMantissa<Real> log_of_2 = LogOf2<DoubleMantissa<Real>>();
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
      return QuietNan<DoubleMantissa<Real>>();
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
    return QuietNan<DoubleMantissa<Real>>();
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

inline float Log2(const float& value) { return std::log2(value); }

inline double Log2(const double& value) { return std::log2(value); }

inline long double Log2(const long double& value) { return std::log2(value); }

template <typename Real>
DoubleMantissa<Real> Log2(const DoubleMantissa<Real>& value) {
  return Log(value) / LogOf2<DoubleMantissa<Real>>();
}

inline float Log10(const float& value) { return std::log10(value); }

inline double Log10(const double& value) { return std::log10(value); }

inline long double Log10(const long double& value) { return std::log10(value); }

template <typename Real>
DoubleMantissa<Real> Log10(const DoubleMantissa<Real>& value) {
  return Log(value) / LogOf10<DoubleMantissa<Real>>();
}

inline float Abs(const float& value) { return std::abs(value); }

inline double Abs(const double& value) { return std::abs(value); }

inline long double Abs(const long double& value) { return std::abs(value); }

template <typename Real>
DoubleMantissa<Real> Abs(const DoubleMantissa<Real>& value) {
  return value.Upper() < Real(0) ? -value : value;
}

template <typename Real>
DoubleMantissa<Real> SinTaylorSeries(const DoubleMantissa<Real>& theta) {
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
                         Epsilon<DoubleMantissa<Real>>().Upper();

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
DoubleMantissa<Real> CosTaylorSeries(const DoubleMantissa<Real>& theta) {
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

  const Real threshold = Real(0.5) * Epsilon<DoubleMantissa<Real>>().Upper();

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
  static const DoubleMantissa<Real> pi = Pi<DoubleMantissa<Real>>();
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
    *sixteenth_pi_remainder = QuietNan<DoubleMantissa<Real>>();
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
    *sixteenth_pi_remainder = QuietNan<DoubleMantissa<Real>>();
#endif  // ifdef MANTIS_DEBUG
  }
}

template <typename Real>
const DoubleMantissa<Real>& SixteenthPiSinTable(int num_sixteenth_pi) {
#ifdef MANTIS_DEBUG
  if (num_sixteenth_pi < 1 || num_sixteenth_pi > 4) {
    std::cerr << "Only multiples of {1, 2, 3, 4} of pi / 16 supported."
              << std::endl;
    return QuietNan<DoubleMantissa<Real>>();
  }
#endif  // ifdef MANTIS_DEBUG
  static const DoubleMantissa<Real> pi = Pi<DoubleMantissa<Real>>();
  static const DoubleMantissa<Real> sixteenth_pi = pi / Real(16);
  static const DoubleMantissa<Real> sin_table[] = {
      SinTaylorSeries(sixteenth_pi), SinTaylorSeries(Real(2) * sixteenth_pi),
      SinTaylorSeries(Real(3) * sixteenth_pi),
      SinTaylorSeries(Real(4) * sixteenth_pi),
  };
  return sin_table[num_sixteenth_pi - 1];
}

template <typename Real>
const DoubleMantissa<Real>& SixteenthPiCosTable(int num_sixteenth_pi) {
#ifdef MANTIS_DEBUG
  if (num_sixteenth_pi < 1 || num_sixteenth_pi > 4) {
    std::cerr << "Only multiples of {1, 2, 3, 4} of pi / 16 supported."
              << std::endl;
    return QuietNan<DoubleMantissa<Real>>();
  }
#endif  // ifdef MANTIS_DEBUG
  static const DoubleMantissa<Real> pi = Pi<DoubleMantissa<Real>>();
  static const DoubleMantissa<Real> sixteenth_pi = pi / Real(16);
  static const DoubleMantissa<Real> cos_table[] = {
      CosTaylorSeries(sixteenth_pi), CosTaylorSeries(Real(2) * sixteenth_pi),
      CosTaylorSeries(Real(3) * sixteenth_pi),
      CosTaylorSeries(Real(4) * sixteenth_pi),
  };
  return cos_table[num_sixteenth_pi - 1];
}

template <typename Real>
DoubleMantissa<Real> Sin(const DoubleMantissa<Real>& theta) {
  if (theta.Upper() == Real(0)) {
    return DoubleMantissa<Real>();
  }

  // We follow the basic approach of the QD library of Hida et al. by
  // decomposing theta into three components:
  //
  //   sin(theta) = sin(k (2 pi) + m (pi / 2) + n (pi / 16) + gamma)
  //              = sin(m (pi / 2) + n (pi / 16) + gamma),
  //
  // where k, m, and n are integers. The contribution of m (pi / 2) is
  // taken care of via branching between {+-sin, +-cos}, and the reduced term,
  //
  //   sin(n (pi / 16) + gamma),
  //
  // is handled via thorough usage of
  //
  //   sin(a + b) = sin(a) cos(b) + cos(a) sin(b),
  //
  // given
  //
  //   a = |num_sixteenth_pi_int * sixteenth_pi| = |n pi / 16| and
  //   b = sixteenth_pi_remainder = gamma.
  //
  // The only components which are not known a priori are sin(b) and cos(b);
  // thanks to |b| <= pi / 16, the Taylor series expansions for sin(b) and
  // cos(b) can be efficiently evaluated.
  //

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  if (num_sixteenth_pi_int == 0) {
    if (num_half_pi_int == 0) {
      // sin(a + b) = sin(b).
      return SinTaylorSeries(sixteenth_pi_remainder);
    } else if (num_half_pi_int == 1) {
      // sin(a + b + (pi / 2)) = sin(b + (pi / 2)) = cos(b).
      return CosTaylorSeries(sixteenth_pi_remainder);
    } else if (num_half_pi_int == -1) {
      // sin(a + b - (pi / 2)) = sin(b - (pi / 2)) = -cos(b).
      return -CosTaylorSeries(sixteenth_pi_remainder);
    } else {
      // sin(a + b + pi) = sin(b + pi) = -sin(b).
      return -SinTaylorSeries(sixteenth_pi_remainder);
    }
  }

  const DoubleMantissa<Real> sin_a =
      SixteenthPiSinTable<Real>(abs_num_sixteenth_pi_int);
  const DoubleMantissa<Real> cos_a =
      SixteenthPiCosTable<Real>(abs_num_sixteenth_pi_int);

  const DoubleMantissa<Real> sin_b = SinTaylorSeries(sixteenth_pi_remainder);
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

  // We follow the basic approach of the QD library of Hida et al. by
  // decomposing theta into three components:
  //
  //   cos(theta) = cos(k (2 pi) + m (pi / 2) + n (pi / 16) + gamma)
  //              = cos(m (pi / 2) + n (pi / 16) + gamma),
  //
  // where k, m, and n are integers. The contribution of m (pi / 2) is
  // taken care of via branching between {+-sin, +-cos}, and the reduced term,
  //
  //   cos(n (pi / 16) + gamma),
  //
  // is handled via thorough usage of
  //
  //   cos(a + b) = cos(a) cos(b) - sin(a) sin(b),
  //
  // given
  //
  //   a = |num_sixteenth_pi_int * sixteenth_pi| = |n pi / 16| and
  //   b = sixteenth_pi_remainder = gamma.
  //
  // The only components which are not known a priori are sin(b) and cos(b);
  // thanks to |b| <= pi / 16, the Taylor series expansions for sin(b) and
  // cos(b) can be efficiently evaluated.
  //

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  if (num_sixteenth_pi_int == 0) {
    if (num_half_pi_int == 0) {
      // cos(a + b) = cos(b).
      return CosTaylorSeries(sixteenth_pi_remainder);
    } else if (num_half_pi_int == 1) {
      // cos(a + b + (pi / 2)) = cos(b + (pi / 2)) = -sin(b).
      return -SinTaylorSeries(sixteenth_pi_remainder);
    } else if (num_half_pi_int == -1) {
      // cos(a + b - (pi / 2)) = cos(b - (pi / 2)) = sin(b).
      return SinTaylorSeries(sixteenth_pi_remainder);
    } else {
      // cos(a + b + pi) = cos(b + pi) = -cos(b).
      return -CosTaylorSeries(sixteenth_pi_remainder);
    }
  }

  const DoubleMantissa<Real> sin_a =
      SixteenthPiSinTable<Real>(abs_num_sixteenth_pi_int);
  const DoubleMantissa<Real> cos_a =
      SixteenthPiCosTable<Real>(abs_num_sixteenth_pi_int);

  const DoubleMantissa<Real> sin_b = SinTaylorSeries(sixteenth_pi_remainder);
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

  // We blend the above decompositional approaches to sin and cos to avoid
  // duplication.

  int num_half_pi_int, num_sixteenth_pi_int;
  DoubleMantissa<Real> sixteenth_pi_remainder;
  DecomposeSinCosArgument(theta, &num_half_pi_int, &num_sixteenth_pi_int,
                          &sixteenth_pi_remainder);
  const int abs_num_sixteenth_pi_int = std::abs(num_sixteenth_pi_int);

  const DoubleMantissa<Real> sin_b = SinTaylorSeries(sixteenth_pi_remainder);
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
  static const DoubleMantissa<Real> pi = Pi<DoubleMantissa<Real>>();
  static const DoubleMantissa<Real> half_pi =
      MultiplyByPowerOfTwo(pi, Real(0.5));
  static const DoubleMantissa<Real> quarter_pi =
      MultiplyByPowerOfTwo(pi, Real(0.25));
  static const DoubleMantissa<Real> three_quarters_pi = Real(3) * quarter_pi;

  if (x.Upper() == Real(0)) {
    if (y.Upper() == Real(0)) {
      // Every value of theta works equally well for the origin.
      return QuietNan<DoubleMantissa<Real>>();
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
    return QuietNan<DoubleMantissa<Real>>();
  }
  if (abs_sin_theta.Upper() == Real(1) && abs_sin_theta.Lower() == Real(0)) {
    static const DoubleMantissa<Real> half_pi =
        MultiplyByPowerOfTwo(Pi<DoubleMantissa<Real>>(), Real(0.5));
    return sin_theta.Upper() > 0 ? half_pi : -half_pi;
  }

  const DoubleMantissa<Real> cos_theta = SinCosDualMagnitude(sin_theta);
  return ArcTan2(sin_theta, cos_theta);
}

template <typename Real>
DoubleMantissa<Real> ArcCos(const DoubleMantissa<Real>& cos_theta) {
  const DoubleMantissa<Real> abs_cos_theta = Abs(cos_theta);
  if (abs_cos_theta.Upper() > Real(1)) {
    return QuietNan<DoubleMantissa<Real>>();
  }
  if (abs_cos_theta.Upper() == Real(1) && abs_cos_theta.Lower() == Real(0)) {
    static const DoubleMantissa<Real> pi = Pi<DoubleMantissa<Real>>();
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
    return QuietNan<DoubleMantissa<Real>>();
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
    return QuietNan<DoubleMantissa<Real>>();
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

}  // namespace mantis

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_MATH_IMPL_H_
