/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_COMPLEX_IMPL_H_
#define MANTIS_COMPLEX_IMPL_H_

#include "mantis/complex.hpp"

namespace mantis {

template <typename RealT>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex()
    MANTIS_NOEXCEPT : std::complex<RealT>() {}

template <typename RealT>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex(
    const Complex<RealT>& input) MANTIS_NOEXCEPT
    : std::complex<RealT>(input.real(), input.imag()) {}

template <typename RealT>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex(
    const std::complex<RealT>& input) MANTIS_NOEXCEPT
    : std::complex<RealT>(input) {}

template <typename RealT>
template <typename RealInputType>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex(
    const RealInputType& input) MANTIS_NOEXCEPT
    : std::complex<RealT>(static_cast<RealT>(input)) {}

template <typename RealT>
template <typename RealInputType>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex(
    const Complex<RealInputType>& input) MANTIS_NOEXCEPT
    : std::complex<RealT>(static_cast<RealT>(input.real()),
                          static_cast<RealT>(input.imag())) {}

template <typename RealT>
template <typename RealInputType, typename ImagInputType>
constexpr Complex<RealT, EnableIf<IsStandardFloat<RealT>>>::Complex(
    const RealInputType& real, const ImagInputType& imag) MANTIS_NOEXCEPT
    : std::complex<RealT>(static_cast<RealT>(real), static_cast<RealT>(imag)) {}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>::Complex() MANTIS_NOEXCEPT {}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>::Complex(
    const DoubleMantissa<RealBase>& real_val,
    const DoubleMantissa<RealBase>& imag_val) MANTIS_NOEXCEPT
    : real_(real_val),
      imag_(imag_val) {}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>::Complex(
    const Complex<DoubleMantissa<RealBase>>& value) MANTIS_NOEXCEPT
    : real_(value.real_),
      imag_(value.imag_) {}

template <typename RealBase>
template <typename RealInputType>
constexpr Complex<DoubleMantissa<RealBase>>::Complex(const RealInputType& input)
    MANTIS_NOEXCEPT : real_(static_cast<RealType>(input)) {}

template <typename RealBase>
template <typename RealInputType>
constexpr Complex<DoubleMantissa<RealBase>>::Complex(
    const Complex<RealInputType>& input) MANTIS_NOEXCEPT
    : real_(static_cast<RealType>(input.Real())),
      imag_(static_cast<RealType>(input.Imag())) {}

template <typename RealBase>
template <typename RealInputType, typename ImagInputType>
constexpr Complex<DoubleMantissa<RealBase>>::Complex(const RealInputType& real,
                                                     const ImagInputType& imag)
    MANTIS_NOEXCEPT : real_(static_cast<RealType>(real)),
                      imag_(static_cast<RealType>(imag)) {}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>& Complex<DoubleMantissa<RealBase>>::
operator=(const Complex<DoubleMantissa<RealBase>>& rhs) MANTIS_NOEXCEPT {
  real_ = rhs.Real();
  imag_ = rhs.Imag();
  return *this;
}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>& Complex<DoubleMantissa<RealBase>>::
operator+=(const Complex<DoubleMantissa<RealBase>>& rhs) MANTIS_NOEXCEPT {
  real_ += rhs.Real();
  imag_ += rhs.Imag();
  return *this;
}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>& Complex<DoubleMantissa<RealBase>>::
operator-=(const Complex<DoubleMantissa<RealBase>>& rhs) MANTIS_NOEXCEPT {
  real_ -= rhs.Real();
  imag_ -= rhs.Imag();
  return *this;
}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>& Complex<DoubleMantissa<RealBase>>::
operator*=(const Complex<DoubleMantissa<RealBase>>& rhs) MANTIS_NOEXCEPT {
  const DoubleMantissa<RealBase> new_real =
      rhs.Real() * real_ - rhs.Imag() * imag_;
  imag_ = rhs.Real() * imag_ + rhs.Imag() * real_;
  real_ = new_real;
  return *this;
}

template <typename RealBase>
constexpr Complex<DoubleMantissa<RealBase>>& Complex<DoubleMantissa<RealBase>>::
operator/=(const Complex<DoubleMantissa<RealBase>>& rhs) {
  Complex<DoubleMantissa<RealBase>> a = *this;
  *this = SmithDiv(a, rhs);
  return *this;
}

template <typename Real>
constexpr bool operator==(const Complex<Real>& lhs, const Complex<Real>& rhs) {
  return lhs.Real() == rhs.Real() && lhs.Imag() == rhs.Imag();
}

template <typename Real>
constexpr bool operator!=(const Complex<Real>& lhs, const Complex<Real>& rhs) {
  return !(lhs == rhs);
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& value)
    MANTIS_NOEXCEPT {
  const std::complex<Real>& value_std =
      static_cast<const std::complex<Real>&>(value);
  return -value_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator-(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return Complex<Real>(-value.Real(), -value.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator+(
    const Complex<Real>& a, const Complex<Real>& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std + b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() + b.Real(), a.Imag() + b.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator+(const Complex<Real>& a,
                                        const Real& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std + b;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() + b, a.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator+(const Real& a, const Complex<Real>& b)
    MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a + b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator+(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a + b.Real(), b.Imag());
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator+(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std + Real(b);
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() + b, a.Imag());
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator+(
    const Integral& a, const Complex<Real>& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return Real(a) + b_std;
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator+(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(b.Real() + a, b.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator-(
    const Complex<Real>& a, const Complex<Real>& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std - b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() - b.Real(), a.Imag() - b.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& a,
                                        const Real& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std - b;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() - b, a.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator-(const Real& a, const Complex<Real>& b)
    MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a - b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator-(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a - b.Real(), -b.Imag());
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator-(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std - Real(b);
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() - b, a.Imag());
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator-(
    const Integral& a, const Complex<Real>& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return Real(a) - b_std;
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator-(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a - b.Real(), -b.Imag());
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std * b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT {
  const Real real = a.Real() * b.Real() - a.Imag() * b.Imag();
  const Real imag = a.Real() * b.Imag() + a.Imag() * b.Real();
  return Complex<Real>(real, imag);
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Real& b)MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std * b;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Real& b)MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() * b, a.Imag() * b);
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator*(const Real& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a * b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator*(const Real& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT {
  return Complex<Real>(a * b.Real(), a * b.Imag());
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator*(const Complex<Real>& a,
                                        const Integral& b)MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std * Real(b);
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Integral& b)MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() * b, a.Imag() * b);
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator*(const Integral& a,
                                        const Complex<Real>& b)MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return Real(a) * b_std;
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator*(const Integral& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT {
  return Complex<Real>(a * b.Real(), a * b.Imag());
}

template <typename Real>
constexpr Complex<Real> NaiveDiv(const Complex<Real>& a,
                                 const Complex<Real>& b) {
  const Real denominator = b.Real() * b.Real() + b.Imag() * b.Imag();
  const Real c_real = (a.Real() * b.Real() + a.Imag() * b.Imag()) / denominator;
  const Real c_imag = (a.Imag() * b.Real() - a.Real() * b.Imag()) / denominator;
  return Complex<Real>(c_real, c_imag);
}

template <typename Real>
constexpr Complex<Real> SmithDiv(const Complex<Real>& a,
                                 const Complex<Real>& b) {
  if (std::abs(b.Imag()) <= std::abs(b.Real())) {
    const Real ratio = b.Imag() / b.Real();
    const Real denominator = b.Real() + b.Imag() * ratio;
    const Real c_real = (a.Real() + a.Imag() * ratio) / denominator;
    const Real c_imag = (a.Imag() - a.Real() * ratio) / denominator;
    return Complex<Real>(c_real, c_imag);
  } else {
    const Real ratio = b.Real() / b.Imag();
    const Real denominator = b.Real() * ratio + b.Imag();
    const Real c_real = (a.Real() * ratio + a.Imag()) / denominator;
    const Real c_imag = (a.Imag() * ratio - a.Real()) / denominator;
    return Complex<Real>(c_real, c_imag);
  }
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a,
                                        const Complex<Real>& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std / b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Complex<Real>& b) {
  return SmithDiv(a, b);
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a, const Real& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std / b;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator/(const Complex<Real>& a, const Real& b) {
  return Complex<Real>(a.Real() / b, a.Imag() / b);
}

template <typename Real, typename>
CXX20_CONSTEXPR Complex<Real> operator/(const Real& a, const Complex<Real>& b) {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a / b_std;
}

template <typename Real, typename, typename>
constexpr Complex<Real> operator/(const Real& a, const Complex<Real>& b) {
  // Run a specialization of Smith's division algorithm.
  if (std::abs(b.Imag()) <= std::abs(b.Real())) {
    const Real ratio = b.Imag() / b.Real();
    const Real denominator = b.Real() + b.Imag() * ratio;
    return Complex<Real>(a / denominator, (-a * ratio) / denominator);
  } else {
    const Real ratio = b.Real() / b.Imag();
    const Real denominator = b.Real() * ratio * b.Imag();
    return Complex<Real>((a * ratio) / denominator, -a / denominator);
  }
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator/(const Complex<Real>& a,
                                        const Integral& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std / Real(b);
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Integral& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() / b, a.Imag() / b);
}

template <typename Real, typename Integral, typename, typename>
CXX20_CONSTEXPR Complex<Real> operator/(
    const Integral& a, const Complex<Real>& b) MANTIS_NOEXCEPT {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return Real(a) / b_std;
}

template <typename Real, typename Integral, typename, typename, typename>
constexpr Complex<Real> operator/(const Integral& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Real(a) / b;
}

template <typename Real, typename>
constexpr Real RealPart(const Real& value) MANTIS_NOEXCEPT {
  return value;
}

template <typename Real, typename>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.real();
}

template <typename Real, typename, typename>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.Real();
}

template <typename Real, typename>
constexpr Real ImagPart(const Real& value) MANTIS_NOEXCEPT {
  return 0;
}

template <typename Real, typename>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.imag();
}

template <typename Real, typename, typename>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.Imag();
}

template <typename Real, typename>
constexpr Real Conjugate(const Real& value) MANTIS_NOEXCEPT {
  return value;
}

template <typename Real>
constexpr Complex<Real> Conjugate(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return Complex<Real>{value.Real(), -value.Imag()};
}

template <typename Real>
constexpr Real Abs(const Complex<Real>& value) {
  return Hypot(value.Real(), value.Imag());
}

template <typename Real, typename>
Complex<Real> SquareRoot(const Complex<Real>& x) {
  return std::sqrt(static_cast<const std::complex<Real>>(x));
}

template <typename Real, typename, typename>
Complex<Real> SquareRoot(const Complex<Real>& x) {
  // TODO(Jack Poulson): Document the branch cuts and formulae.
  if (x.Real() == 0) {
    // The pure imaginary branch.
    const Real tau = SquareRoot(std::abs(x.Imag()) / 2);
    const Real real = tau;
    const Real imag = x.Imag() >= 0 ? tau : -tau;
    return Complex<Real>(real, imag);
  } else {
    const Real tau = SquareRoot(2 * (Abs(x) + std::abs(x.Real())));
    const Real half_tau = tau / 2;
    if (x.Real() > 0) {
      return Complex<Real>(half_tau, x.Imag() / tau);
    } else {
      const Real real = std::abs(x.Imag()) / tau;
      const Real imag = x.Imag() >= 0 ? half_tau : -half_tau;
      return Complex<Real>(real, imag);
    }
  }
}

template <typename Real>
Real Arg(const Complex<Real>& x) {
  return std::atan2(x.Imag(), x.Real());
}

template <typename Real, typename>
Complex<Real> Log(const Complex<Real>& x) {
  return std::log(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> Log(const Complex<Real>& x) {
  return Complex<Real>(Log(Abs(x)), Arg(x));
}

template <typename Real>
Complex<Real> Log2(const Complex<Real>& x) {
  return Log(x) / LogOf2<Real>();
}

template <typename Real, typename>
Complex<Real> Log10(const Complex<Real>& x) {
  return std::log10(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> Log10(const Complex<Real>& x) {
  return Log(x) / LogOf10<Real>();
}

template <typename Real, typename>
Complex<Real> Sin(const Complex<Real>& x) {
  return std::sin(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> Sin(const Complex<Real>& x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  return Complex<Real>(real_sin * imag_cosh, real_cos * imag_sinh);
}

template <typename Real, typename>
Complex<Real> Cos(const Complex<Real>& x) {
  return std::cos(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> Cos(const Complex<Real>& x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  return Complex<Real>(real_cos * imag_cosh, -real_sin * imag_sinh);
}

template <typename Real, typename>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x,
            Complex<Real>* cos_x) {
  *sin_x = std::sin(x);
  *cos_x = std::cos(x);
}

template <typename Real, typename, typename>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x,
            Complex<Real>* cos_x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  *sin_x = Complex<Real>(real_sin * imag_cosh, real_cos * imag_sinh);
  *cos_x = Complex<Real>(real_cos * imag_cosh, -real_sin * imag_sinh);
}

template <typename Real, typename>
Complex<Real> Tan(const Complex<Real>& x) {
  return std::tan(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> Tan(const Complex<Real>& x) {
  Complex<Real> sin_x, cos_x;
  SinCos(x, &sin_x, &cos_x);
  return sin_x / cos_x;
}

template <typename Real, typename>
Complex<Real> ArcSin(const Complex<Real>& x) {
  return std::asin(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcSin(const Complex<Real>& x) {
  // Arcsin(x) = -i Arcsinh(i x)
  const Complex<Real> y = ArcHyperbolicSin(Complex<Real>(-x.Imag(), x.Real()));
  return Complex<Real>(y.Imag(), -y.Real());
}

template <typename Real, typename>
Complex<Real> ArcCos(const Complex<Real>& x) {
  return std::acos(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcCos(const Complex<Real>& x) {
  static const Real pi = std::acos(Real(-1));
  static const Real half_pi = pi / Real(2);
  const Complex<Real> asin_x = ArcSin(x);
  return Complex<Real>(half_pi - asin_x.Real(), -asin_x.Imag());
}

template <typename Real, typename>
Complex<Real> ArcTan(const Complex<Real>& x) {
  return std::atan(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcTan(const Complex<Real>& x) {
  // The solution follows that of [Abramowitz/Stegun-1972]:
  //
  //   atan(x) = (1 / 2i) Ln((i - x) / (i + x)).
  //
  // Defining z = (i - x) / (i + x),
  //
  //   atan(x) = (1 / 2i) Ln(z)
  //           = (1 / 2i) (Ln(|z|) + i arg(z))
  //           = (1 / 2) (arg(z) - i Ln(|z|))
  //           = (1 / 2) (arg(z) + i Ln(1 / |z|).
  //
  // To compute the inputs to the atan2 used to compute arg, we can ignore the
  // denominator of the textbook division algorithm for (i + x) / (i - x):
  //
  //   real_num = -Re[x]^2 + (1 - Im[x])(1 + Im[x])
  //            = 1 - Re[x]^2 - Im[x]^2,
  //
  //   imag_num = Re[x] (1 - Im[x]) + (1 + Im[x]) Re[x]
  //            = 2 Re[x].
  //
  // The component Ln(|z|) can be decomposed as:
  //
  //   Ln(1 / |z|) = (1 / 2) Ln(1 / |z|^2)
  //               = (1 / 2) (Ln(|i + x|^2) - Ln(|i - x|^2)).
  //
  const Real real_square = x.Real() * x.Real();
  const Real imag_square = x.Imag() * x.Imag();

  // Compute the real component of the solution.
  const Real atan_x_real =
      ArcTan2(Real(2) * x.Real(), Real(1) - real_square - imag_square) / 2;

  // Compute |i + x|^2.
  const Real log_numerator =
      real_square + (Real(1) + x.Imag()) * (Real(1) + x.Imag());

  // Compute |i - x|^2.
  const Real log_denominator =
      real_square + (Real(1) - x.Imag()) * (Real(1) - x.Imag());

  // Compute the imaginary components of the solution.
  const Real atan_x_imag = (Log(log_numerator) - Log(log_denominator)) / 4;

  return Complex<Real>(atan_x_real, atan_x_imag);
}

template <typename Real, typename>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x) {
  return std::asinh(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x) {
  // The solution follows that of [Abramowitz/Stegun-1972]:
  //
  // Arcsinh(x) = Ln(x + sqrt(1 + x * x)) = Ln(x + tau),
  //
  // where
  //
  // tau^2 := 1 + x * x
  //        = 1 + (R[x] + I[x] i) (R[x] + I[x] i)
  //        = 1 + (R[x]^2 - I[x]^2) + 2 R[x] I[x] i
  //        = 1 + (R[x] - I[x])(R[x] + I[x]) + 2 R[x] I[x] i.
  //
  // yields the solution Ln(x + tau).
  Complex<Real> tau(Real(1) + (x.Real() - x.Imag()) * (x.Real() + x.Imag()),
                    Real(2) * x.Real() * x.Imag());
  tau = SquareRoot(tau);
  return Log(tau + x);
}

template <typename Real, typename>
Complex<Real> ArcHyperbolicCos(const Complex<Real>& x) {
  return std::acosh(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcHyperbolicCos(const Complex<Real>& x) {
  // The solution follows that of [Abramowitz/Stegun-1972]:
  //
  // Arcsinh(x) = Ln(x + sqrt(x * x - 1)) = Ln(x + tau),
  //
  // where
  //
  // tau^2 := x * x - 1
  //        = (R[x] + I[x] i) (R[x] + I[x] i) - 1
  //        = (R[x]^2 - I[x]^2) + 2 R[x] I[x] i - 1
  //        = [(R[x] - I[x])(R[x] + I[x]) - 1] + [2 R[x] I[x]] i.
  //
  // yields the solution Ln(x + tau).
  Complex<Real> tau((x.Real() - x.Imag()) * (x.Real() + x.Imag()) - Real(1),
                    Real(2) * x.Real() * x.Imag());
  tau = SquareRoot(tau);
  return Log(tau + x);
}

template <typename Real, typename>
Complex<Real> ArcHyperbolicTan(const Complex<Real>& x) {
  return std::atanh(static_cast<const std::complex<Real>&>(x));
}

template <typename Real, typename, typename>
Complex<Real> ArcHyperbolicTan(const Complex<Real>& x) {
  // The solution follows that of [Abramowitz/Stegun-1972]:
  //
  // Arctanh(x) = (1 / 2) Ln((1 + x) / (1 - x)).
  //
  // And Ln(z) = (Ln(|z|), arg(z)).
  //
  // Thus, the real component is
  //
  //    (1 / 2) Ln(|(1 + x) / (1 - x)|) =
  //    (1 / 4) Ln(|(1 + x) / (1 - x)|^2) =
  //    (1 / 4) (Ln(|1 + x|^2) - Ln(|1 - x|^2)).
  //
  // The imaginary component is
  //
  //    (1 / 2) arg((1 + x) / (1 - x)).
  //
  // We can ignore the denominator of the real and imaginary components of the
  // textbook division formula for (1 + x) / (1 - x) to find the atan2
  // arguments:
  //
  //     real_num = (1 + Re[x])(1 - Re[x]) - Im[x]^2
  //              = 1 - Re[x]^2 - Im[x]^2,
  //
  //     imag_num = Im[x] * (1 - Re[x]) + (1 + Re[x]) * Im[x]
  //              = 2 Im[x].
  //
  const Real real_square = x.Real() * x.Real();
  const Real imag_square = x.Imag() * x.Imag();

  // Compute |1 + x|^2.
  const Real log_numerator =
      (Real(1) + x.Real()) * (Real(1) + x.Real()) + imag_square;

  // Compute |1 - x|^2.
  const Real log_denominator =
      (Real(1) - x.Real()) * (Real(1) - x.Real()) + imag_square;

  // Compute the real components of the solution.
  const Real atanh_x_real = (Log(log_numerator) - Log(log_denominator)) / 4;

  // Compute the imaginary component of the solution.
  const Real atanh_x_imag =
      ArcTan2(Real(2) * x.Imag(), Real(1) - real_square - imag_square) / 2;

  return Complex<Real>(atanh_x_real, atanh_x_imag);
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const Complex<Real>& value) {
  out << RealPart(value) << " + " << ImagPart(value) << "i";
  return out;
}

// References:
//
// [Abramowitz/Stegun-1972]
//   Milton Abramowitz and Irene A. Stegun,
//   "Handbook of Mathematical Functions, Dover Pub., New York, 1972.
//

}  // namespace mantis

namespace std {

template <typename Real>
string to_string(const mantis::Complex<Real>& value) {
  return std::to_string(value.Real()) + " + " + std::to_string(value.Imag()) +
         "i";
}

template <typename Real, typename>
Real real(const Real& x) {
  return x;
}

template <typename Real>
Real real(const mantis::Complex<Real>& x) {
  return x.Real();
}

template <typename Real, typename>
Real imag(const Real& x) {
  return 0;
}

template <typename Real>
Real imag(const mantis::Complex<Real>& x) {
  return x.Imag();
}

template <typename Real, typename>
mantis::Complex<Real> conj(const Real& x) {
  return mantis::Complex<Real>(x);
}

template <typename Real>
mantis::Complex<Real> conj(const mantis::Complex<Real>& x) {
  return mantis::Conjugate(x);
}

template <typename Real>
Real abs(const mantis::Complex<Real>& x) {
  return mantis::Abs(x);
}

template <typename Real>
Real arg(const mantis::Complex<Real>& x) {
  return mantis::Arg(x);
}

template <typename Real>
mantis::Complex<Real> sqrt(const mantis::Complex<Real>& x) {
  return mantis::SquareRoot(x);
}

template <typename Real>
mantis::Complex<Real> log(const mantis::Complex<Real>& x) {
  return mantis::Log(x);
}

template <typename Real>
mantis::Complex<Real> log2(const mantis::Complex<Real>& x) {
  return mantis::Log2(x);
}

template <typename Real>
mantis::Complex<Real> log10(const mantis::Complex<Real>& x) {
  return mantis::Log10(x);
}

template <typename Real>
mantis::Complex<Real> cos(const mantis::Complex<Real>& x) {
  return mantis::Cos(x);
}

template <typename Real>
mantis::Complex<Real> sin(const mantis::Complex<Real>& x) {
  return mantis::Sin(x);
}

template <typename Real>
mantis::Complex<Real> tan(const mantis::Complex<Real>& x) {
  return mantis::Tan(x);
}

template <typename Real>
mantis::Complex<Real> acos(const mantis::Complex<Real>& x) {
  return mantis::ArcCos(x);
}

template <typename Real>
mantis::Complex<Real> asin(const mantis::Complex<Real>& x) {
  return mantis::ArcSin(x);
}

template <typename Real>
mantis::Complex<Real> atan(const mantis::Complex<Real>& x) {
  return mantis::ArcTan(x);
}

template <typename Real>
mantis::Complex<Real> cosh(const mantis::Complex<Real>& x) {
  return mantis::HyperbolicCos(x);
}

template <typename Real>
mantis::Complex<Real> sinh(const mantis::Complex<Real>& x) {
  return mantis::HyperbolicSin(x);
}

template <typename Real>
mantis::Complex<Real> tanh(const mantis::Complex<Real>& x) {
  return mantis::HyperbolicTan(x);
}

template <typename Real>
mantis::Complex<Real> acosh(const mantis::Complex<Real>& x) {
  return mantis::ArcHyperbolicCos(x);
}

template <typename Real>
mantis::Complex<Real> asinh(const mantis::Complex<Real>& x) {
  return mantis::ArcHyperbolicSin(x);
}

template <typename Real>
mantis::Complex<Real> atanh(const mantis::Complex<Real>& x) {
  return mantis::ArcHyperbolicTan(x);
}

}  // namespace std

#endif  // ifndef MANTIS_COMPLEX_IMPL_H_
