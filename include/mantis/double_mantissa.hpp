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
#include <type_traits>

namespace mantis {

// For overloading function definitions using type traits. For example:
//
//   template<typename T, typename=EnableIf<std::is_integral<T>>>
//   int Log(T value);
//
//   template<typename T, typename=DisableIf<std::is_integral<T>>>
//   double Log(T value);
//
// would lead to the 'Log' function returning an 'int' for any integral type
// and a 'double' for any non-integral type.
template <typename Condition, class T = void>
using EnableIf = typename std::enable_if<Condition::value, T>::type;

template <typename Condition, class T = void>
using DisableIf = typename std::enable_if<!Condition::value, T>::type;

// A structure for the decimal scientific notation representation of a
// floating-point value. It makes use of the standard scientific notation:
//
//   +- d_0 . d_1 d_2 ... d_{n - 1} x 10^{exponent},
//
// where each d_j lives in [0, 9] and 'exponent' is an integer.
// We contiguously store the decimal digits with this implied format. For
// example, to represent the first several digits of pi, we would have:
//
//   positive: true,
//   exponent: 0,
//   d: [3, 1, 4, 1, 5, 9, ...]
//
// To represent -152.4, we would have
//
//   positive: false,
//   exponent: 2,
//   d: [1, 5, 2, 4]
//
// The special value of NaN is handled via:
//
//   positive: true,
//   exponent: 0,
//   d: ['n', 'a', 'n'].
//
// Similarly, infinity is handled via:
//
//   positive: true,
//   exponent: 0,
//   d: ['i', 'n', 'f'],
//
// and negative infinity has 'positive' equal to false.
//
struct ScientificNotation {
  // The sign of the value.
  bool positive = true;

  // The exponent of the scientific notation of the value.
  int exponent = 0;

  // Each entry contains a value in the range 0 to 9, except in the cases of
  // NaN and +-infinity.
  std::vector<unsigned char> digits;

  // Returns a string for the decimal scientific notation.
  std::string ToString() const;

  // Fills this class by converting a string in decimal scientific notation.
  ScientificNotation& FromString(const std::string& rep);
};

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
  // Create a typedef to the underlying base representation.
  typedef Real Base;

  // Initializes the double-mantissa scalar to zero.
  constexpr DoubleMantissa();

  // Initializes the double-mantissa scalar to the given single-mantissa value.
  constexpr DoubleMantissa(const Real& upper);

  // Constructs the double-mantissa scalar given two single-mantissa values.
  constexpr DoubleMantissa(const Real& upper, const Real& lower);

  // Copy constructor.
  DoubleMantissa(const DoubleMantissa<Real>& value);

  // Constructs from the decimal scientific notation.
  DoubleMantissa(const ScientificNotation& rep);

  // Constructs from a string decimal representation.
  DoubleMantissa(const std::string& rep);

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

  // Returns true if the individual components are equivalent.
  bool operator==(const DoubleMantissa<Real>& value) const;

  // Returns true if the individual components vary.
  bool operator!=(const DoubleMantissa<Real>& value) const;

  // Returns true if this class's data is less than the given value.
  bool operator<(const DoubleMantissa<Real>& value) const;

  // Returns true if this class's data is <= than the given value.
  bool operator<=(const DoubleMantissa<Real>& value) const;

  // Returns true if this class's data is greater than the given value.
  bool operator>(const DoubleMantissa<Real>& value) const;

  // Returns true if this class's data is >= than the given value.
  bool operator>=(const DoubleMantissa<Real>& value) const;

  // Return the flooring of the value into a int.
  operator int() const;

  // Return the flooring of the value into a long int.
  operator long int() const;

  // Return the flooring of the value into a long long int.
  operator long long int() const;

  // Casts the double-mantissa value into a float.
  operator float() const;

  // Casts the double-mantissa value into a double.
  operator double() const;

  // Casts the double-mantissa value into a long double.
  operator long double() const;

  // Convert the double-mantissa value into scientific notation.
  ScientificNotation ToScientificNotation(int num_digits) const;

  // Fills this double-mantissa value using a decimal scientific representation.
  DoubleMantissa<Real>& FromScientificNotation(const ScientificNotation& rep);

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
struct IsDoubleMantissa {
  static const bool value = false;
};

template <typename Real>
struct IsDoubleMantissa<DoubleMantissa<Real>> {
  static const bool value = true;
};

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
Real LogMax();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
Real LogOf2();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
Real LogOf10();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
Real Pi();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
Real EulerNumber();

namespace double_mantissa {

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
constexpr DoubleMantissa<Real> Epsilon();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
constexpr DoubleMantissa<Real> Infinity();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
constexpr DoubleMantissa<Real> QuietNan();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
constexpr DoubleMantissa<Real> SignalingNan();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> LogOf2();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> ComputeLogOf2();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> LogOf10();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> Pi();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> ComputePi();

template <typename Real, typename = DisableIf<IsDoubleMantissa<Real>>>
DoubleMantissa<Real> EulerNumber();

}  // namespace double_mantissa

// Returns the absolute value of the double-mantissa value.
// Like the other routines, it is assumed that the value is reduced.
template <typename Real>
DoubleMantissa<Real> Abs(const DoubleMantissa<Real>& value);

// Returns the inverse of an extended-precision value.
template <typename Real>
DoubleMantissa<Real> Inverse(const DoubleMantissa<Real>& value);

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

// Returns value ^ exponent, where exponent is integer-valued.
template <typename Real>
DoubleMantissa<Real> IntegerPower(const DoubleMantissa<Real>& value,
                                  int exponent);

// Returns value ^ exponent, where exponent is also a double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Power(const DoubleMantissa<Real>& value,
                           const DoubleMantissa<Real>& exponent);

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

// Fills the sine and cosine of the given double-mantissa value.
template <typename Real>
void SinCos(const DoubleMantissa<Real>& value, DoubleMantissa<Real>* s,
            DoubleMantissa<Real>* c);

// Returns the tangent of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> Tan(const DoubleMantissa<Real>& value);

// Returns the inverse tangent of the double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcTan(const DoubleMantissa<Real>& tan_theta);

// Returns the two-argument inverse tangent of the double-mantissa value.
// The result of theta = arctan2(y, x) is such that, for some r > 0,
//   x = r cos(theta), y = r sin(theta).
template <typename Real>
DoubleMantissa<Real> ArcTan2(const DoubleMantissa<Real>& y,
                             const DoubleMantissa<Real>& x);

// Returns the inverse cosine of the double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcCos(const DoubleMantissa<Real>& cos_theta);

// Returns the inverse sine of the double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcSin(const DoubleMantissa<Real>& sin_theta);

// Returns the hyperbolic sine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> HyperbolicSin(const DoubleMantissa<Real>& x);

// Returns the hyperbolic cosine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> HyperbolicCos(const DoubleMantissa<Real>& x);

// Returns the hyperbolic tangent of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> HyperbolicTan(const DoubleMantissa<Real>& x);

// Returns the inverse hyperbolic sine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcHyperbolicSin(const DoubleMantissa<Real>& sinh_x);

// Returns the inverse hyperbolic cosine of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcHyperbolicCos(const DoubleMantissa<Real>& cosh_x);

// Returns the inverse hyperbolic tangent of the given double-mantissa value.
template <typename Real>
DoubleMantissa<Real> ArcHyperbolicTan(const DoubleMantissa<Real>& tanh_x);

// Returns the two-norm of (x, y) in a manner which avoids unnecessary underflow
// or overflow. The replacement of the naive computation is
//     r = sqrt(x^2 + y^2)
//       = |max_abs(x, y)| sqrt(1 + (min_abs(x, y) / max_abs(x, y))^2).
template <typename Real>
DoubleMantissa<Real> Hypot(const DoubleMantissa<Real>& x,
                           const DoubleMantissa<Real>& y);

// Returns the rounding of a double-mantissa value to the floored integer.
template <typename Real>
DoubleMantissa<Real> Floor(const DoubleMantissa<Real>& value);

// Returns the rounding of a double-mantissa value to the nearest integer.
template <typename Real>
DoubleMantissa<Real> Round(const DoubleMantissa<Real>& value);

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
mantis::DoubleMantissa<Real> abs(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> acos(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> acosh(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> asin(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> asinh(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> atan(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> atan2(const mantis::DoubleMantissa<Real>& y,
                                   const mantis::DoubleMantissa<Real>& x);

template <typename Real>
mantis::DoubleMantissa<Real> atanh(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> cos(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> cosh(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> exp(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> floor(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
bool isfinite(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
bool isinf(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
bool isnan(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> ldexp(const mantis::DoubleMantissa<Real>& value,
                                   int exp);

template <typename Real>
mantis::DoubleMantissa<Real> log(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> log10(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> pow(const mantis::DoubleMantissa<Real>& value,
                                 int exponent);

template <typename Real>
mantis::DoubleMantissa<Real> pow(const mantis::DoubleMantissa<Real>& value,
                                 const mantis::DoubleMantissa<Real>& exponent);

template <typename Real>
mantis::DoubleMantissa<Real> round(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> sin(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> sinh(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> sqrt(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> tan(const mantis::DoubleMantissa<Real>& value);

template <typename Real>
mantis::DoubleMantissa<Real> tanh(const mantis::DoubleMantissa<Real>& value);

}  // namespace std

#include "mantis/double_mantissa-impl.hpp"

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_H_
