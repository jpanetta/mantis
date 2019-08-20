/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_COMPLEX_H_
#define MANTIS_COMPLEX_H_

#include <complex>
#include <type_traits>

#include "mantis/double_mantissa.hpp"
#include "mantis/enable_if.hpp"
#include "mantis/macros.hpp"

namespace mantis {

// The structure's value being 'true' should equate with the ability to
// safely template std::complex over it.
template <typename Real>
struct IsStandardFloat {
  static const bool value = false;
};

template <>
struct IsStandardFloat<float> {
  static const bool value = true;
};
template <>
struct IsStandardFloat<double> {
  static const bool value = true;
};
template <>
struct IsStandardFloat<long double> {
  static const bool value = true;
};

// An extension of std::complex beyond {float, double, long double}.
// Unfortunately, manually specializing std::complex to floating-point types
// outside of this set, such as DoubleMantissa, often does not allow routines
// such as the std::complex version of std::exp to properly resolve calls to
// real math functions such as std::exp(DoubleMantissa).
template <typename RealT, typename = void>
class Complex {
 public:
  // The underlying real type of the complex class.
  typedef RealT RealType;

  constexpr Complex() MANTIS_NOEXCEPT {}

  constexpr Complex(const Complex<RealType>& value) MANTIS_NOEXCEPT
      : real_(value.Real()),
        imag_(value.Imag()) {}

  constexpr Complex(const RealType& real_val,
                    const RealType& imag_val) MANTIS_NOEXCEPT
      : real_(real_val),
        imag_(imag_val) {}

  constexpr Complex(const RealType& real_val) MANTIS_NOEXCEPT
      : real_(real_val) {}

  template <typename Integral, typename = EnableIf<std::is_integral<Integral>>>
  constexpr Complex(const Integral& real_val) MANTIS_NOEXCEPT
      : real_(real_val) {}

  constexpr const RealType real() const MANTIS_NOEXCEPT { return real_; }
  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real_; }

  constexpr const RealType imag() const MANTIS_NOEXCEPT { return imag_; }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag_; }

 private:
  // The real and imaginary components of the complex variable.
  RealType real_, imag_;
};

// A specialization of Complex to an underlying real type which can be used to
// instantiate an std::complex.
template <typename RealT>
class Complex<RealT, EnableIf<IsStandardFloat<RealT>>>
    : public std::complex<RealT> {
 public:
  // The underlying real type of the complex class.
  typedef RealT RealType;
  typedef std::complex<RealType> Base;

  // Imports of the std::complex operators.
  using Base::operator=;
  using Base::operator+=;
  using Base::operator-=;
  using Base::operator*=;
  using Base::operator/=;

  // The default constructor.
  constexpr Complex() MANTIS_NOEXCEPT;

  // A copy constructor from a Complex<Real> variable.
  constexpr Complex(const Complex<RealType>& input) MANTIS_NOEXCEPT;

  // A copy constructor from a std::complex variable.
  constexpr Complex(const std::complex<RealType>& input) MANTIS_NOEXCEPT;

  // A copy constructor from a real variable.
  template <typename RealInputType>
  constexpr Complex(const RealInputType& input) MANTIS_NOEXCEPT;

  // A copy constructor from real and imaginary parts.
  template <typename RealInputType, typename ImagInputType>
  constexpr Complex(const RealInputType& real,
                    const ImagInputType& imag) MANTIS_NOEXCEPT;

  // A copy constructor from a Complex variable.
  template <typename RealInputType>
  constexpr Complex(const Complex<RealInputType>& input) MANTIS_NOEXCEPT;

  constexpr const RealType Real() const MANTIS_NOEXCEPT { return Base::real(); }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return Base::imag(); }
};

// A partial specialization of Complex to an underlying real type of
// DoubleMantissa<Real>.
template <typename RealBase>
class Complex<DoubleMantissa<RealBase>> {
 public:
  typedef DoubleMantissa<RealBase> RealType;

  constexpr Complex() MANTIS_NOEXCEPT;

  constexpr Complex(const RealType& real_val,
                    const RealType& imag_val) MANTIS_NOEXCEPT;

  constexpr Complex(const Complex<RealType>& value) MANTIS_NOEXCEPT;

  // A copy constructor from a real variable.
  template <typename RealInputType>
  constexpr Complex(const RealInputType& input) MANTIS_NOEXCEPT;

  // A copy constructor from real and imaginary parts.
  template <typename RealInputType, typename ImagInputType>
  constexpr Complex(const RealInputType& real,
                    const ImagInputType& imag) MANTIS_NOEXCEPT;

  // A copy constructor from a Complex variable.
  template <typename RealInputType>
  constexpr Complex(const Complex<RealInputType>& input) MANTIS_NOEXCEPT;

  constexpr Complex<RealType>& operator=(const Complex<RealType>& rhs)
      MANTIS_NOEXCEPT;
  constexpr Complex<RealType>& operator+=(const Complex<RealType>& rhs)
      MANTIS_NOEXCEPT;
  constexpr Complex<RealType>& operator-=(const Complex<RealType>& rhs)
      MANTIS_NOEXCEPT;
  constexpr Complex<RealType>& operator*=(const Complex<RealType>& rhs)
      MANTIS_NOEXCEPT;
  constexpr Complex<RealType>& operator/=(const Complex<RealType>& rhs);

  constexpr const RealType real() const MANTIS_NOEXCEPT { return real_; }
  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real_; }
  constexpr const RealType imag() const MANTIS_NOEXCEPT { return imag_; }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag_; }

 private:
  RealType real_, imag_;
};

template <typename Real>
constexpr bool operator==(const Complex<Real>& lhs, const Complex<Real>& rhs);
template <typename Real>
constexpr bool operator!=(const Complex<Real>& lhs, const Complex<Real>& rhs);

namespace complex_base {

template <typename Real>
struct ComplexBaseHelper {
  typedef Real type;
};

template <typename Real>
struct ComplexBaseHelper<Complex<Real>> {
  typedef Real type;
};

}  // namespace complex_base

// Returns the type of the base field of a real or complex scalar. For example:
//   ComplexBase<double> == double
//   ComplexBase<Complex<double>> == double.
template <typename Field>
using ComplexBase = typename complex_base::ComplexBaseHelper<Field>::type;

// Encodes whether or not a given type is complex. For example,
//   IsComplex<double>::value == false
//   IsComplex<Complex<double>>::value == true
template <typename Real>
struct IsComplex {
  static constexpr bool value = false;
};

template <typename Real>
struct IsComplex<Complex<Real>> {
  static constexpr bool value = true;
};

// Encodes whether or not a given type is real. For example,
//   IsComplex<double>::value == true
//   IsComplex<Complex<double>>::value == false
template <typename Field>
struct IsReal {
  static constexpr bool value = !IsComplex<Field>::value;
};

// Returns the negation of a complex value.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& value)
    MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator-(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the sum of two values.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator+(const Complex<Real>& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator+(const Complex<Real>& a,
                                        const Real& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator+(const Real& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator+(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator+(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator+(const Integral& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator+(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

// Returns the difference of two values.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& a,
                                        const Real& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Real& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator-(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator-(const Integral& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator-(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

// Returns the product of two values.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Real& b)MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Real& b)MANTIS_NOEXCEPT;

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator*(const Real& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator*(const Real& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Integral& b)MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Integral& b)MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator*(const Integral& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator*(const Integral& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT;

// Returns a / b using the naive, textbook algorithm. This approach is used by
// GCC.
template <typename Real>
constexpr Complex<Real> NaiveDiv(const Complex<Real>& a,
                                 const Complex<Real>& b);

// Returns a / b using "Smith's algorithm", which is Fig. 3 from Baudin and
// Smith. This approach is typically more accurate than the naive approach and
// is our default for operator/, despite the increase in cost relative to
// GCC's adoption of the naive algorithm for std::complex operator/.
template <typename Real>
constexpr Complex<Real> SmithDiv(const Complex<Real>& a,
                                 const Complex<Real>& b);

// TODO(Jack Poulson): Add support for a 'safe' division algorithm which avoids
// unnecessary overflow or underflow.

// Returns the ratio of two values.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a,
                                        const Complex<Real>& b);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Complex<Real>& b);

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a, const Real& b);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator/(const Complex<Real>& a, const Real& b);

template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
CXX20_CONSTEXPR Complex<Real> operator/(const Real& a, const Complex<Real>& b);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Complex<Real> operator/(const Real& a, const Complex<Real>& b);

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT;

template <typename Real, typename Integral,
          typename = EnableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>>
CXX20_CONSTEXPR Complex<Real> operator/(const Integral& a,
                                        const Complex<Real>& b) MANTIS_NOEXCEPT;
template <typename Real, typename Integral,
          typename = DisableIf<IsStandardFloat<Real>>,
          typename = EnableIf<std::is_integral<Integral>>, typename = void>
constexpr Complex<Real> operator/(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

// Returns the real part of a real scalar.
template <typename Real, typename = DisableIf<IsComplex<Real>>>
constexpr Real RealPart(const Real& value) MANTIS_NOEXCEPT;

// Returns the real part of a complex scalar.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the imaginary part of a real scalar (zero).
template <typename Real, typename = DisableIf<IsComplex<Real>>>
constexpr Real ImagPart(const Real& value) MANTIS_NOEXCEPT;

// Returns the imaginary part of a complex scalar.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT;
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the complex-conjugate of a real value (the value itself).
template <typename Real, typename = DisableIf<IsComplex<Real>>>
constexpr Real Conjugate(const Real& value) MANTIS_NOEXCEPT;

// Returns the complex-conjugate of a complex value.
template <typename Real>
constexpr Complex<Real> Conjugate(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the magnitude of a complex value.
template <typename Real>
constexpr Real Abs(const Complex<Real>& value);

// Returns the square-root of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> SquareRoot(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> SquareRoot(const Complex<Real>& x);

// Returns the argument of a complex number.
template <typename Real>
Real Arg(const Complex<Real>& x);

// Returns the natural logarithm of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> Log(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> Log(const Complex<Real>& x);

// Returns the (base two) logarithm of a complex number.
// NOTE: There is no std::complex std::log2.
template <typename Real>
Complex<Real> Log2(const Complex<Real>& x);

// Returns the (base ten) logarithm of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> Log10(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> Log10(const Complex<Real>& x);

// Returns the sine of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> Sin(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> Sin(const Complex<Real>& x);

// Returns the cosine of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> Cos(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> Cos(const Complex<Real>& x);

// Simultaneously computes the sin and cosine of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x, Complex<Real>* cos_x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x, Complex<Real>* cos_x);

// Returns the tangent of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> Tan(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> Tan(const Complex<Real>& x);

// Returns the inverse sine of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcSin(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcSin(const Complex<Real>& x);

// Returns the inverse cosine of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcCos(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcCos(const Complex<Real>& x);

// Returns the inverse tangent of the complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcTan(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcTan(const Complex<Real>& x);

// Returns the inverse hyperbolic sine of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x);

// Returns the inverse hyperbolic cosine of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcHyperbolicCos(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcHyperbolicCos(const Complex<Real>& x);

// Returns the inverse hyperbolic tangent of a complex number.
template <typename Real, typename = EnableIf<IsStandardFloat<Real>>>
Complex<Real> ArcHyperbolicTan(const Complex<Real>& x);
template <typename Real, typename = DisableIf<IsStandardFloat<Real>>,
          typename = void>
Complex<Real> ArcHyperbolicTan(const Complex<Real>& x);

// Pretty-prints the complex value.
template <typename Real>
std::ostream& operator<<(std::ostream& out, const Complex<Real>& value);

}  // namespace mantis

namespace std {

template <typename Real>
string to_string(const mantis::Complex<Real>& value);

template <typename Real,
          typename = mantis::EnableIf<mantis::IsDoubleMantissa<Real>>>
Real real(const Real& x);

template <typename Real>
Real real(const mantis::Complex<Real>& x);

template <typename Real,
          typename = mantis::EnableIf<mantis::IsDoubleMantissa<Real>>>
Real imag(const Real& x);

template <typename Real>
Real imag(const mantis::Complex<Real>& x);

template <typename Real,
          typename = mantis::EnableIf<mantis::IsDoubleMantissa<Real>>>
mantis::Complex<Real> conj(const Real& x);

template <typename Real>
mantis::Complex<Real> conj(const mantis::Complex<Real>& x);

template <typename Real>
Real abs(const mantis::Complex<Real>& x);

template <typename Real>
Real arg(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> sqrt(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> log(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> log2(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> log10(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> cos(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> sin(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> tan(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> acos(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> asin(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> atan(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> cosh(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> sinh(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> tanh(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> acosh(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> asinh(const mantis::Complex<Real>& x);

template <typename Real>
mantis::Complex<Real> atanh(const mantis::Complex<Real>& x);

}  // namespace std

#include "mantis/complex-impl.hpp"

#endif  // ifndef MANTIS_COMPLEX_H_
