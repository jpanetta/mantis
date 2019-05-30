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

// An extension of std::complex beyond {float, double, long double}.
// Unfortunately, manually specializing std::complex to floating-point types
// outside of this set, such as DoubleMantissa, often does not allow routines
// such as the std::complex version of std::exp to properly resolve calls to
// real math functions such as std::exp(DoubleMantissa).
template <typename RealType>
class Complex {
 public:
  constexpr Complex(const RealType& real_val,
                    const RealType& imag_val) MANTIS_NOEXCEPT
      : real_(real_val),
        imag_(imag_val) {}

  constexpr const RealType real() const MANTIS_NOEXCEPT { return real_; }
  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real_; }

  constexpr const RealType imag() const MANTIS_NOEXCEPT { return imag_; }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag_; }

 private:
  // The real and imaginary components of the complex variable.
  RealType real_, imag_;
};

// A specialization of Complex to an underlying real type of 'float'.
template <>
class Complex<float> : public std::complex<float> {
 public:
  // The underlying real type of the complex class.
  typedef float RealType;

  // Imports of the std::complex operators.
  using std::complex<RealType>::operator=;
  using std::complex<RealType>::operator+=;
  using std::complex<RealType>::operator-=;
  using std::complex<RealType>::operator*=;
  using std::complex<RealType>::operator/=;

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
  template <typename RealInputType, class ImagInputType>
  constexpr Complex(const RealInputType& real,
                    const ImagInputType& imag) MANTIS_NOEXCEPT;

  // A copy constructor from a Complex variable.
  template <typename RealInputType>
  constexpr Complex(const Complex<RealInputType>& input) MANTIS_NOEXCEPT;

  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real(); }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag(); }
};

// A specialization of Complex to an underlying real type of 'double'.
template <>
class Complex<double> : public std::complex<double> {
 public:
  // The underlying real type of the complex class.
  typedef double RealType;

  // Imports of the std::complex operators.
  using std::complex<RealType>::operator=;
  using std::complex<RealType>::operator+=;
  using std::complex<RealType>::operator-=;
  using std::complex<RealType>::operator*=;
  using std::complex<RealType>::operator/=;

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
  template <typename RealInputType, class ImagInputType>
  constexpr Complex(const RealInputType& real,
                    const ImagInputType& imag) MANTIS_NOEXCEPT;

  // A copy constructor from a Complex variable.
  template <typename RealInputType>
  constexpr Complex(const Complex<RealInputType>& input) MANTIS_NOEXCEPT;

  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real(); }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag(); }
};

// A specialization of Complex to an underlying real type of 'long double'.
template <>
class Complex<long double> : public std::complex<long double> {
 public:
  // The underlying real type of the complex class.
  typedef long double RealType;

  // Imports of the std::complex operators.
  using std::complex<RealType>::operator=;
  using std::complex<RealType>::operator+=;
  using std::complex<RealType>::operator-=;
  using std::complex<RealType>::operator*=;
  using std::complex<RealType>::operator/=;

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
  template <typename RealInputType, class ImagInputType>
  constexpr Complex(const RealInputType& real,
                    const ImagInputType& imag) MANTIS_NOEXCEPT;

  // A copy constructor from a Complex variable.
  template <typename RealInputType>
  constexpr Complex(const Complex<RealInputType>& input) MANTIS_NOEXCEPT;

  constexpr const RealType Real() const MANTIS_NOEXCEPT { return real(); }
  constexpr const RealType Imag() const MANTIS_NOEXCEPT { return imag(); }
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
template <class Field>
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
template <class Field>
struct IsReal {
  static constexpr bool value = !IsComplex<Field>::value;
};

// Returns the negation of a complex value.
CXX20_CONSTEXPR Complex<float> operator-(const Complex<float>& value)
    MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator-(const Complex<double>& value)
    MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator-(
    const Complex<long double>& value) MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the sum of two values.
CXX20_CONSTEXPR Complex<float> operator+(
    const Complex<float>& a, const Complex<float>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator+(
    const Complex<double>& a, const Complex<double>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator+(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator+(const Complex<float>& a,
                                         const float& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator+(const Complex<double>& a,
                                          const double& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator+(
    const Complex<long double>& a, const long double& b) MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator+(
    const float& a, const Complex<float>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator+(
    const double& a, const Complex<double>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator+(
    const long double& a, const Complex<long double>& b) MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator+(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

// Returns the difference of two values.
CXX20_CONSTEXPR Complex<float> operator-(
    const Complex<float>& a, const Complex<float>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator-(
    const Complex<double>& a, const Complex<double>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator-(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator-(const Complex<float>& a,
                                         const float& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator-(const Complex<double>& a,
                                          const double& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator-(
    const Complex<long double>& a, const long double& b) MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator-(
    const float& a, const Complex<float>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator-(
    const double& a, const Complex<double>& b) MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator-(
    const long double& a, const Complex<long double>& b) MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator-(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT;

// Returns the product of two values.
CXX20_CONSTEXPR Complex<float> operator*(
    const Complex<float>& a, const Complex<float>& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator*(
    const Complex<double>& a, const Complex<double>& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator*(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator*(const Complex<float>& a,
                                         const float& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator*(const Complex<double>& a,
                                          const double& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator*(
    const Complex<long double>& a, const long double& b)MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Real& b)MANTIS_NOEXCEPT;

CXX20_CONSTEXPR Complex<float> operator*(
    const float& a, const Complex<float>& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<double> operator*(
    const double& a, const Complex<double>& b)MANTIS_NOEXCEPT;
CXX20_CONSTEXPR Complex<long double> operator*(
    const long double& a, const Complex<long double>& b)MANTIS_NOEXCEPT;
template <typename Real>
constexpr Complex<Real> operator*(const Real& a,
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
CXX20_CONSTEXPR Complex<float> operator/(const Complex<float>& a,
                                         const Complex<float>& b);
CXX20_CONSTEXPR Complex<double> operator/(const Complex<double>& a,
                                          const Complex<double>& b);
CXX20_CONSTEXPR Complex<long double> operator/(const Complex<long double>& a,
                                               const Complex<long double>& b);
template <typename Real>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Complex<Real>& b);

CXX20_CONSTEXPR Complex<float> operator/(const Complex<float>& a,
                                         const float& b);
CXX20_CONSTEXPR Complex<double> operator/(const Complex<double>& a,
                                          const double& b);
CXX20_CONSTEXPR Complex<long double> operator/(const Complex<long double>& a,
                                               const long double& b);
template <typename Real>
constexpr Complex<Real> operator/(const Complex<Real>& a, const Real& b);

CXX20_CONSTEXPR Complex<float> operator/(const float& a,
                                         const Complex<float>& b);
CXX20_CONSTEXPR Complex<double> operator/(const double& a,
                                          const Complex<double>& b);
CXX20_CONSTEXPR Complex<long double> operator/(const long double& a,
                                               const Complex<long double>& b);
template <typename Real>
constexpr Complex<Real> operator/(const Real& a, const Complex<Real>& b);

// Returns the real part of a real scalar.
template <typename Real>
constexpr Real RealPart(const Real& value) MANTIS_NOEXCEPT;

// Returns the real part of a complex scalar.
constexpr float RealPart(const Complex<float>& value) MANTIS_NOEXCEPT;
constexpr double RealPart(const Complex<double>& value) MANTIS_NOEXCEPT;
constexpr long double RealPart(const Complex<long double>& value)
    MANTIS_NOEXCEPT;
template <typename Real>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the imaginary part of a real scalar (zero).
template <typename Real>
constexpr Real ImagPart(const Real& value) MANTIS_NOEXCEPT;

// Returns the imaginary part of a complex scalar.
constexpr float ImagPart(const Complex<float>& value) MANTIS_NOEXCEPT;
constexpr double ImagPart(const Complex<double>& value) MANTIS_NOEXCEPT;
constexpr long double ImagPart(const Complex<long double>& value)
    MANTIS_NOEXCEPT;
template <typename Real>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the complex-conjugate of a real value (the value itself).
template <typename Real>
constexpr Real Conjugate(const Real& value) MANTIS_NOEXCEPT;

// Returns the complex-conjugate of a complex value.
template <typename Real>
constexpr Complex<Real> Conjugate(const Complex<Real>& value) MANTIS_NOEXCEPT;

// Returns the magnitude of a complex value.
template <typename Real>
constexpr Real Abs(const Complex<Real>& value);

// Returns the square-root of a complex number.
Complex<float> SquareRoot(const Complex<float>& x);
Complex<double> SquareRoot(const Complex<double>& x);
Complex<long double> SquareRoot(const Complex<long double>& x);
template <typename Real>
Complex<Real> SquareRoot(const Complex<Real>& x);

// Returns the argument of a complex number.
template <typename Real>
Real Arg(const Complex<Real>& x);

// Returns the natural logarithm of a complex number.
Complex<float> Log(const Complex<float>& x);
Complex<double> Log(const Complex<double>& x);
Complex<long double> Log(const Complex<long double>& x);
template <typename Real>
Complex<Real> Log(const Complex<Real>& x);

// Returns the sine of the complex number.
Complex<float> Sin(const Complex<float>& x);
Complex<double> Sin(const Complex<double>& x);
Complex<long double> Sin(const Complex<long double>& x);
template <typename Real>
Complex<Real> Sin(const Complex<Real>& x);

// Returns the cosine of the complex number.
Complex<float> Cos(const Complex<float>& x);
Complex<double> Cos(const Complex<double>& x);
Complex<long double> Cos(const Complex<long double>& x);
template <typename Real>
Complex<Real> Cos(const Complex<Real>& x);

// Simultaneously computes the sin and cosine of a complex number.
void SinCos(const Complex<float>& x, Complex<float>* sin_x,
            Complex<float>* cos_x);
void SinCos(const Complex<double>& x, Complex<double>* sin_x,
            Complex<double>* cos_x);
void SinCos(const Complex<long double>& x, Complex<long double>* sin_x,
            Complex<long double>* cos_x);
template <typename Real>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x, Complex<Real>* cos_x);

// Returns the tangent of the complex number.
Complex<float> Tan(const Complex<float>& x);
Complex<double> Tan(const Complex<double>& x);
Complex<long double> Tan(const Complex<long double>& x);
template <typename Real>
Complex<Real> Tan(const Complex<Real>& x);

// Returns the tangent of the complex number.
Complex<float> ArcTan(const Complex<float>& x);
Complex<double> ArcTan(const Complex<double>& x);
Complex<long double> ArcTan(const Complex<long double>& x);
template <typename Real>
Complex<Real> ArcTan(const Complex<Real>& x);

// Returns the inverse hyperbolic sine of a complex number.
Complex<float> ArcHyperbolicSin(const Complex<float>& x);
Complex<double> ArcHyperbolicSin(const Complex<double>& x);
Complex<long double> ArcHyperbolicSin(const Complex<long double>& x);
template <typename Real>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x);

// Returns the inverse hyperbolic cosine of a complex number.
Complex<float> ArcHyperbolicCos(const Complex<float>& x);
Complex<double> ArcHyperbolicCos(const Complex<double>& x);
Complex<long double> ArcHyperbolicCos(const Complex<long double>& x);
template <typename Real>
Complex<Real> ArcHyperbolicCos(const Complex<Real>& x);

// Returns the inverse hyperbolic tangent of a complex number.
Complex<float> ArcHyperbolicTan(const Complex<float>& x);
Complex<double> ArcHyperbolicTan(const Complex<double>& x);
Complex<long double> ArcHyperbolicTan(const Complex<long double>& x);
template <typename Real>
Complex<Real> ArcHyperbolicTan(const Complex<Real>& x);

// Pretty-prints the complex value.
template <typename Real>
std::ostream& operator<<(std::ostream& out, const Complex<Real>& value);

}  // namespace mantis

#include "mantis/complex-impl.hpp"

#endif  // ifndef MANTIS_COMPLEX_H_
