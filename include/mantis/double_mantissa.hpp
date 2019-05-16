/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_H_
#define MANTIS_DOUBLE_MANTISSA_H_

#include <cmath>
#include <limits>

namespace mantis {

// A class which concatenates the mantissas of two real floating-point values
// in order to produce a datatype whose mantissa is twice the length of one of
// the two components. The typical case is for the underlying datatype to be
// an IEEE 64-bit floating-point representation (double).
//
// Our primary reference for the implementation is:
//
//   Yozo Hida, Xiaoye S. Li, and David H. Bailey,
//   "Library for Double-Double and Quad-Double Arithmetic",
//   Technical Report, NERSC Division, Lawrence Berkeley National Laboratory,
//   USA, 2007.
//
// Said paper in turn frequently cites from:
//
//   Jonathan R. Shewchuk,
//   "Adaptive precision floating-point arithmetic and fast robust geometric
//    predicates.", Discrete & Computational Geometry, 18(3):305--363, 1997.
//
template <typename Real>
class DoubleMantissa {
 public:
  // Initializes the double-mantissa scalar to zero.
  constexpr DoubleMantissa();

  // Initializes the double-mantissa scalar to the given single-mantissa value.
  constexpr DoubleMantissa(const Real& upper);

  // Constructs the double-mantissa scalar given two single-mantissa values.
  constexpr DoubleMantissa(const Real& upper, const Real& lower);

  // Copy constructor.
  DoubleMantissa(const DoubleMantissa<Real>& value);

  // Returns a reference to the larger half of the double-mantissa value.
  Real& Upper();

  // Returns a const reference to the larger half of the double-mantissa value.
  const Real& Upper() const;

  // Returns a reference to the smaller half of the double-mantissa value.
  Real& Lower();

  // Returns a const reference to the smaller half of the double-mantissa value.
  const Real& Lower() const;

  // Overwrites the internal representation to contain as much information as
  // possible in the upper component.
  DoubleMantissa<Real>& Reduce();

  // The assignment operator.
  DoubleMantissa<Real>& operator=(const Real& value);

  // The assignment operator.
  DoubleMantissa<Real>& operator=(const DoubleMantissa<Real>& value);

  // Adds a single-mantissa value to the current state.
  DoubleMantissa<Real>& operator+=(const Real& value);

  // Adds a double-mantissa value to the current state.
  DoubleMantissa<Real>& operator+=(const DoubleMantissa<Real>& value);

  // Subtracts a single-mantissa value to the current state.
  DoubleMantissa<Real>& operator-=(const Real& value);

  // Subtracts a double-mantissa value to the current state.
  DoubleMantissa<Real>& operator-=(const DoubleMantissa<Real>& value);

  // Multiplies the current state by a single-mantissa value.
  DoubleMantissa<Real>& operator*=(const Real& value);

  // Multiplies the current state by a double-mantissa value.
  DoubleMantissa<Real>& operator*=(const DoubleMantissa<Real>& value);

  // Divides the current state by a single-mantissa value.
  DoubleMantissa<Real>& operator/=(const Real& value);

  // Divides the current state by a double-mantissa value.
  DoubleMantissa<Real>& operator/=(const DoubleMantissa<Real>& value);

  // Checks if the individual components are equivalent.
  bool operator==(const DoubleMantissa<Real>& value) const;

  // Checks if the individual components vary.
  bool operator!=(const DoubleMantissa<Real>& value) const;

  // TODO(Jack Poulson):
  // operator int() const;

  // TODO(Jack Poulson):
  // operator long int() const;

  // TODO(Jack Poulson):
  // operator long long int() const;

  // Casts the double-mantissa value into a float.
  operator float() const;

  // Casts the double-mantissa value into a double.
  operator double() const;

  // Casts the double-mantissa value into a long double.
  operator long double() const;

  // Returns the approximate ratio x / y using one refinement.
  static DoubleMantissa<Real> FastDivide(const DoubleMantissa<Real>& x,
                                         const DoubleMantissa<Real>& y);

  // Returns the approximate ratio x / y using two refinements.
  static DoubleMantissa<Real> Divide(const DoubleMantissa<Real>& x,
                                     const DoubleMantissa<Real>& y);

 private:
  // The upper, followed by lower, contributions to the double-mantissa value.
  Real values_[2];
};

template <typename Real>
Real LogMax();

template <typename Real>
Real LogOf2();

template <typename Real>
Real LogOf10();

template <typename Real>
Real Pi();

template <typename Real>
Real EulerNumber();

namespace double_mantissa {

template <typename Real>
constexpr DoubleMantissa<Real> Epsilon();

template <typename Real>
constexpr DoubleMantissa<Real> Infinity();

template <typename Real>
constexpr DoubleMantissa<Real> QuietNan();

template <typename Real>
constexpr DoubleMantissa<Real> SignalingNan();

template <typename Real>
DoubleMantissa<Real> LogOf2();

template <typename Real>
DoubleMantissa<Real> ComputeLogOf2();

template <typename Real>
DoubleMantissa<Real> LogOf10();

template <typename Real>
DoubleMantissa<Real> Pi();

template <typename Real>
DoubleMantissa<Real> ComputePi();

template <typename Real>
DoubleMantissa<Real> EulerNumber();

}  // namespace double_mantissa

// Returns the square of an extended-precision value.
template <typename Real>
DoubleMantissa<Real> Square(const DoubleMantissa<Real>& value);

// Returns the square-root of an extended-precision value.
template <typename Real>
DoubleMantissa<Real> SquareRoot(const DoubleMantissa<Real>& value);

// A faster algorithm for multiplying a double-mantissa value by a
// single-mantissa value that is known to be an integer power of two.
template <typename Real>
DoubleMantissa<Real> MultiplyByPowerOfTwo(const DoubleMantissa<Real>& value,
                                          const Real& power_of_two);

// An equivalent of the routine std::ldexp ("Load Exponent") for
// double-mantissa values. The result is value * 2^exp.
template <typename Real>
DoubleMantissa<Real> LoadExponent(const DoubleMantissa<Real>& value, int exp);

// Returns the exponential of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Exp(const DoubleMantissa<Real>& value);

// Returns the (natural) log of the given positive double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Log(const DoubleMantissa<Real>& value);

// Returns the log base-10 of the given positive double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Log10(const DoubleMantissa<Real>& value);

// Returns the sine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Sin(const DoubleMantissa<Real>& value);

// Returns the cosine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Cos(const DoubleMantissa<Real>& value);

}  // namespace mantis

// Returns the negation of the extended-precision value.
template <typename Real>
mantis::DoubleMantissa<Real> operator-(
    const mantis::DoubleMantissa<Real>& value);

// Returns the sum of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator+(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the sum of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator+(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y);

// Returns the sum of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator+(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the difference between the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator-(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the difference between the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator-(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y);

// Returns the difference between the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator-(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the product of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator*(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the product of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator*(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y);

// Returns the product of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator*(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the ratio of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator/(const Real& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Returns the ratio of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator/(const mantis::DoubleMantissa<Real>& x,
                                       const Real& y);

// Returns the ratio of the two values.
template <typename Real>
mantis::DoubleMantissa<Real> operator/(const mantis::DoubleMantissa<Real>& x,
                                       const mantis::DoubleMantissa<Real>& y);

// Pretty-prints the extended-precision value.
template <typename Real>
std::ostream& operator<<(std::ostream& out,
                         const mantis::DoubleMantissa<Real>& value);

namespace std {

// We specialize std::numeric_limits for each of the standard concrete instances
// of DoubleMantissa<Real>, where Real is either float, double, or long double.
//
// We recall that IEEE compliance is lost, exponent properties are conserved,
// and the number of digits in the significand/mantissa is doubled.

// A specialization of std::numeric_limits for DoubleMantissa<float>.
template <>
class numeric_limits<mantis::DoubleMantissa<float>> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = numeric_limits<float>::has_infinity;
  static constexpr bool has_quiet_NaN = numeric_limits<float>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN =
      numeric_limits<float>::has_signaling_NaN;
  static constexpr bool has_denorm = numeric_limits<float>::has_denorm;
  static constexpr bool has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr std::float_round_style round_style =
      numeric_limits<float>::round_style;

  // The extension loses IEEE compliance.
  static constexpr bool is_iec559 = false;

  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr int digits = 2 * numeric_limits<float>::digits;
  static constexpr int digits10 = (digits - 1) * log10(2);
  static constexpr int max_digits10 = ceil(digits * log10(2) + 1);
  static constexpr int radix = numeric_limits<float>::radix;

  static constexpr int min_exponent = numeric_limits<float>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<float>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<float>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<float>::max_exponent10;

  static constexpr bool traps = numeric_limits<float>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr mantis::DoubleMantissa<float> epsilon();
  static constexpr mantis::DoubleMantissa<float> infinity();
  static constexpr mantis::DoubleMantissa<float> quiet_NaN();
  static constexpr mantis::DoubleMantissa<float> signaling_NaN();
};

// A specialization of std::numeric_limits for DoubleMantissa<double>.
template <>
class numeric_limits<mantis::DoubleMantissa<double>> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = numeric_limits<double>::has_infinity;
  static constexpr bool has_quiet_NaN = numeric_limits<double>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN =
      numeric_limits<double>::has_signaling_NaN;
  static constexpr bool has_denorm = numeric_limits<double>::has_denorm;
  static constexpr bool has_denorm_loss =
      numeric_limits<double>::has_denorm_loss;
  static constexpr std::float_round_style round_style =
      numeric_limits<double>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr int digits = 2 * numeric_limits<double>::digits;
  static constexpr int digits10 = (digits - 1) * log10(2);
  static constexpr int max_digits10 = ceil(digits * log10(2) + 1);
  static constexpr int radix = numeric_limits<double>::radix;

  static constexpr int min_exponent = numeric_limits<double>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<double>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<double>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<double>::max_exponent10;

  static constexpr bool traps = numeric_limits<double>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<double>::tinyness_before;

  static constexpr mantis::DoubleMantissa<double> epsilon();
  static constexpr mantis::DoubleMantissa<double> infinity();
  static constexpr mantis::DoubleMantissa<double> quiet_NaN();
  static constexpr mantis::DoubleMantissa<double> signaling_NaN();
};

// A specialization of std::numeric_limits for DoubleMantissa<long double>.
template <>
class numeric_limits<mantis::DoubleMantissa<long double>> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity =
      numeric_limits<long double>::has_infinity;
  static constexpr bool has_quiet_NaN =
      numeric_limits<long double>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN =
      numeric_limits<long double>::has_signaling_NaN;
  static constexpr bool has_denorm = numeric_limits<long double>::has_denorm;
  static constexpr bool has_denorm_loss =
      numeric_limits<long double>::has_denorm_loss;
  static constexpr std::float_round_style round_style =
      numeric_limits<long double>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr int digits = 2 * numeric_limits<long double>::digits;
  static constexpr int digits10 = (digits - 1) * log10(2);
  static constexpr int max_digits10 = ceil(digits * log10(2) + 1);
  static constexpr int radix = numeric_limits<long double>::radix;

  static constexpr int min_exponent = numeric_limits<long double>::min_exponent;
  static constexpr int min_exponent10 =
      numeric_limits<long double>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<long double>::max_exponent;
  static constexpr int max_exponent10 =
      numeric_limits<long double>::max_exponent10;

  static constexpr bool traps = numeric_limits<long double>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<long double>::tinyness_before;

  static constexpr mantis::DoubleMantissa<long double> epsilon();
  static constexpr mantis::DoubleMantissa<long double> infinity();
  static constexpr mantis::DoubleMantissa<long double> quiet_NaN();
  static constexpr mantis::DoubleMantissa<long double> signaling_NaN();
};

template <typename Real>
mantis::DoubleMantissa<Real> cos(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> exp(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> ldexp(const mantis::DoubleMantissa<Real>& value,
                                   int exp);

template <typename Real>
mantis::DoubleMantissa<Real> log(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> log10(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> sin(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> sqrt(const mantis::DoubleMantissa<Real>& value);

}  // namespace std

#include "mantis/double_mantissa-impl.hpp"

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_H_
