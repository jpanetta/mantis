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

constexpr Complex<float>::Complex() MANTIS_NOEXCEPT : std::complex<float>() {}

constexpr Complex<double>::Complex() MANTIS_NOEXCEPT : std::complex<double>() {}

constexpr Complex<long double>::Complex() MANTIS_NOEXCEPT
    : std::complex<long double>() {}

constexpr Complex<float>::Complex(const Complex<float>& input) MANTIS_NOEXCEPT
    : std::complex<float>(input.real(), input.imag()) {}

constexpr Complex<double>::Complex(const Complex<double>& input) MANTIS_NOEXCEPT
    : std::complex<double>(input.real(), input.imag()) {}

constexpr Complex<long double>::Complex(const Complex<long double>& input)
    MANTIS_NOEXCEPT : std::complex<long double>(input.real(), input.imag()) {}

constexpr Complex<float>::Complex(const std::complex<float>& input)
    MANTIS_NOEXCEPT : std::complex<float>(input) {}

constexpr Complex<double>::Complex(const std::complex<double>& input)
    MANTIS_NOEXCEPT : std::complex<double>(input) {}

constexpr Complex<long double>::Complex(const std::complex<long double>& input)
    MANTIS_NOEXCEPT : std::complex<long double>(input) {}

template <class RealInputType>
constexpr Complex<float>::Complex(const RealInputType& input) MANTIS_NOEXCEPT
    : std::complex<float>(static_cast<float>(input)) {}

template <class RealInputType>
constexpr Complex<double>::Complex(const RealInputType& input) MANTIS_NOEXCEPT
    : std::complex<double>(static_cast<double>(input)) {}

template <class RealInputType>
constexpr Complex<long double>::Complex(const RealInputType& input)
    MANTIS_NOEXCEPT
    : std::complex<long double>(static_cast<long double>(input)) {}

template <class RealInputType>
constexpr Complex<float>::Complex(const Complex<RealInputType>& input)
    MANTIS_NOEXCEPT : std::complex<float>(static_cast<float>(input.real()),
                                          static_cast<float>(input.imag())) {}

template <class RealInputType>
constexpr Complex<double>::Complex(const Complex<RealInputType>& input)
    MANTIS_NOEXCEPT : std::complex<double>(static_cast<double>(input.real()),
                                           static_cast<double>(input.imag())) {}

template <class RealInputType>
constexpr Complex<long double>::Complex(const Complex<RealInputType>& input)
    MANTIS_NOEXCEPT
    : std::complex<long double>(static_cast<long double>(input.real()),
                                static_cast<long double>(input.imag())) {}

template <class RealInputType, class ImagInputType>
constexpr Complex<float>::Complex(const RealInputType& real,
                                  const ImagInputType& imag) MANTIS_NOEXCEPT
    : std::complex<float>(static_cast<float>(real), static_cast<float>(imag)) {}

template <class RealInputType, class ImagInputType>
constexpr Complex<double>::Complex(const RealInputType& real,
                                   const ImagInputType& imag) MANTIS_NOEXCEPT
    : std::complex<double>(static_cast<double>(real),
                           static_cast<double>(imag)) {}

template <class RealInputType, class ImagInputType>
constexpr Complex<long double>::Complex(
    const RealInputType& real, const ImagInputType& imag) MANTIS_NOEXCEPT
    : std::complex<long double>(static_cast<long double>(real),
                                static_cast<long double>(imag)) {}

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

CXX20_CONSTEXPR Complex<float> operator-(const Complex<float>& value)
    MANTIS_NOEXCEPT {
  const std::complex<float>& value_std =
      static_cast<const std::complex<float>&>(value);
  return -value_std;
}

CXX20_CONSTEXPR Complex<double> operator-(const Complex<double>& value)
    MANTIS_NOEXCEPT {
  const std::complex<double>& value_std =
      static_cast<const std::complex<double>&>(value);
  return -value_std;
}

CXX20_CONSTEXPR Complex<long double> operator-(
    const Complex<long double>& value) MANTIS_NOEXCEPT {
  const std::complex<long double>& value_std =
      static_cast<const std::complex<long double>&>(value);
  return -value_std;
}

template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return Complex<Real>(-value.Real(), -value.Imag());
}

CXX20_CONSTEXPR Complex<float> operator+(
    const Complex<float>& a, const Complex<float>& b) MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a_std + b_std;
}

CXX20_CONSTEXPR Complex<double> operator+(
    const Complex<double>& a, const Complex<double>& b) MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a_std + b_std;
}

CXX20_CONSTEXPR Complex<long double> operator+(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a_std + b_std;
}

template <typename Real>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() + b.Real(), a.Imag() + b.Imag());
}

CXX20_CONSTEXPR Complex<float> operator+(const Complex<float>& a,
                                         const float& b) MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  return a_std + b;
}

CXX20_CONSTEXPR Complex<double> operator+(const Complex<double>& a,
                                          const double& b) MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  return a_std + b;
}

CXX20_CONSTEXPR Complex<long double> operator+(
    const Complex<long double>& a, const long double& b) MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  return a_std + b;
}

template <typename Real>
constexpr Complex<Real> operator+(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() + b, a.Imag());
}

CXX20_CONSTEXPR Complex<float> operator+(
    const float& a, const Complex<float>& b) MANTIS_NOEXCEPT {
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a + b_std;
}

CXX20_CONSTEXPR Complex<double> operator+(
    const double& a, const Complex<double>& b) MANTIS_NOEXCEPT {
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a + b_std;
}

CXX20_CONSTEXPR Complex<long double> operator+(
    const long double& a, const Complex<long double>& b) MANTIS_NOEXCEPT {
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a + b_std;
}

template <typename Real>
constexpr Complex<Real> operator+(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a + b.Real(), b.Imag());
}

CXX20_CONSTEXPR Complex<float> operator-(
    const Complex<float>& a, const Complex<float>& b) MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a_std - b_std;
}

CXX20_CONSTEXPR Complex<double> operator-(
    const Complex<double>& a, const Complex<double>& b) MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a_std - b_std;
}

CXX20_CONSTEXPR Complex<long double> operator-(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a_std - b_std;
}

template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() - a.Imag(), b.Real() - b.Imag());
}

CXX20_CONSTEXPR Complex<float> operator-(const Complex<float>& a,
                                         const float& b) MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  return a_std - b;
}

CXX20_CONSTEXPR Complex<double> operator-(const Complex<double>& a,
                                          const double& b) MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  return a_std - b;
}

CXX20_CONSTEXPR Complex<long double> operator-(
    const Complex<long double>& a, const long double& b) MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  return a_std - b;
}

template <typename Real>
constexpr Complex<Real> operator-(const Complex<Real>& a,
                                  const Real& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() - b, a.Imag());
}

CXX20_CONSTEXPR Complex<float> operator-(
    const float& a, const Complex<float>& b) MANTIS_NOEXCEPT {
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a - b_std;
}

CXX20_CONSTEXPR Complex<double> operator-(
    const double& a, const Complex<double>& b) MANTIS_NOEXCEPT {
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a - b_std;
}

CXX20_CONSTEXPR Complex<long double> operator-(
    const long double& a, const Complex<long double>& b) MANTIS_NOEXCEPT {
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a - b_std;
}

template <typename Real>
constexpr Complex<Real> operator-(const Real& a,
                                  const Complex<Real>& b) MANTIS_NOEXCEPT {
  return Complex<Real>(a - b.Real(), -b.Imag());
}

CXX20_CONSTEXPR Complex<float> operator*(
    const Complex<float>& a, const Complex<float>& b)MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a_std * b_std;
}

CXX20_CONSTEXPR Complex<double> operator*(
    const Complex<double>& a, const Complex<double>& b)MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a_std * b_std;
}

CXX20_CONSTEXPR Complex<long double> operator*(const Complex<long double>& a,
                                               const Complex<long double>& b)
    MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a_std * b_std;
}

template <typename Real>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Complex<Real>& b)MANTIS_NOEXCEPT {
  const Real real = a.Real() * b.Real() - a.Imag() * b.Imag();
  const Real imag = a.Real() * b.Imag() + a.Imag() * b.Real();
  return Complex<Real>(real, imag);
}

CXX20_CONSTEXPR Complex<float> operator*(const Complex<float>& a,
                                         const float& b)MANTIS_NOEXCEPT {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  return a_std * b;
}

CXX20_CONSTEXPR Complex<double> operator*(const Complex<double>& a,
                                          const double& b)MANTIS_NOEXCEPT {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  return a_std * b;
}

CXX20_CONSTEXPR Complex<long double> operator*(
    const Complex<long double>& a, const long double& b)MANTIS_NOEXCEPT {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  return a_std * b;
}

template <typename Real>
constexpr Complex<Real> operator*(const Complex<Real>& a,
                                  const Real& b)MANTIS_NOEXCEPT {
  return Complex<Real>(a.Real() * b, a.Imag() * b);
}

CXX20_CONSTEXPR Complex<float> operator*(
    const float& a, const Complex<float>& b)MANTIS_NOEXCEPT {
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a * b_std;
}

CXX20_CONSTEXPR Complex<double> operator*(
    const double& a, const Complex<double>& b)MANTIS_NOEXCEPT {
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a * b_std;
}

CXX20_CONSTEXPR Complex<long double> operator*(
    const long double& a, const Complex<long double>& b)MANTIS_NOEXCEPT {
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a * b_std;
}

template <typename Real>
constexpr Complex<Real> operator*(const Real& a,
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

CXX20_CONSTEXPR Complex<float> operator/(const Complex<float>& a,
                                         const Complex<float>& b) {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a_std / b_std;
}

CXX20_CONSTEXPR Complex<double> operator/(const Complex<double>& a,
                                          const Complex<double>& b) {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a_std / b_std;
}

CXX20_CONSTEXPR Complex<long double> operator/(const Complex<long double>& a,
                                               const Complex<long double>& b) {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a_std / b_std;
}

template <typename Real>
constexpr Complex<Real> operator/(const Complex<Real>& a,
                                  const Complex<Real>& b) {
  return SmithDiv(a, b);
}

CXX20_CONSTEXPR Complex<float> operator/(const Complex<float>& a,
                                         const float& b) {
  const std::complex<float>& a_std = static_cast<const std::complex<float>&>(a);
  return a_std / b;
}

CXX20_CONSTEXPR Complex<double> operator/(const Complex<double>& a,
                                          const double& b) {
  const std::complex<double>& a_std =
      static_cast<const std::complex<double>&>(a);
  return a_std / b;
}

CXX20_CONSTEXPR Complex<long double> operator/(const Complex<long double>& a,
                                               const long double& b) {
  const std::complex<long double>& a_std =
      static_cast<const std::complex<long double>&>(a);
  return a_std / b;
}

template <typename Real>
constexpr Complex<Real> operator/(const Complex<Real>& a, const Real& b) {
  return Complex<Real>(a.Real() / b, a.Imag() / b);
}

CXX20_CONSTEXPR Complex<float> operator/(const float& a,
                                         const Complex<float>& b) {
  const std::complex<float>& b_std = static_cast<const std::complex<float>&>(b);
  return a / b_std;
}

CXX20_CONSTEXPR Complex<double> operator/(const double& a,
                                          const Complex<double>& b) {
  const std::complex<double>& b_std =
      static_cast<const std::complex<double>&>(b);
  return a / b_std;
}

CXX20_CONSTEXPR Complex<long double> operator/(const long double& a,
                                               const Complex<long double>& b) {
  const std::complex<long double>& b_std =
      static_cast<const std::complex<long double>&>(b);
  return a / b_std;
}

template <typename Real>
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

template <typename Real>
constexpr Real RealPart(const Real& value) MANTIS_NOEXCEPT {
  return value;
}

constexpr float RealPart(const Complex<float>& value) MANTIS_NOEXCEPT {
  return value.real();
}

constexpr double RealPart(const Complex<double>& value) MANTIS_NOEXCEPT {
  return value.real();
}

constexpr long double RealPart(const Complex<long double>& value)
    MANTIS_NOEXCEPT {
  return value.real();
}

template <typename Real>
constexpr Real RealPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.Real();
}

template <typename Real>
constexpr Real ImagPart(const Real& value) MANTIS_NOEXCEPT {
  return 0;
}

constexpr float ImagPart(const Complex<float>& value) MANTIS_NOEXCEPT {
  return value.imag();
}

constexpr double ImagPart(const Complex<double>& value) MANTIS_NOEXCEPT {
  return value.imag();
}

constexpr long double ImagPart(const Complex<long double>& value)
    MANTIS_NOEXCEPT {
  return value.imag();
}

template <typename Real>
constexpr Real ImagPart(const Complex<Real>& value) MANTIS_NOEXCEPT {
  return value.Imag();
}

template <typename Real>
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

inline Complex<float> SquareRoot(const Complex<float>& x) {
  return std::sqrt(x);
}

inline Complex<double> SquareRoot(const Complex<double>& x) {
  return std::sqrt(x);
}

inline Complex<long double> SquareRoot(const Complex<long double>& x) {
  return std::sqrt(x);
}

template <typename Real>
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

inline Complex<float> Log(const Complex<float>& x) { return std::log(x); }

inline Complex<double> Log(const Complex<double>& x) { return std::log(x); }

inline Complex<long double> Log(const Complex<long double>& x) {
  return std::log(x);
}

template <typename Real>
Complex<Real> Log(const Complex<Real>& x) {
  return Complex<Real>(Log(Abs(x)), Arg(x));
}

inline Complex<float> Sin(const Complex<float>& x) { return std::sin(x); }

inline Complex<double> Sin(const Complex<double>& x) { return std::sin(x); }

inline Complex<long double> Sin(const Complex<long double>& x) {
  return std::sin(x);
}

template <typename Real>
Complex<Real> Sin(const Complex<Real>& x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  return Complex<Real>(real_sin * imag_cosh, real_cos * imag_sinh);
}

inline Complex<float> Cos(const Complex<float>& x) { return std::cos(x); }

inline Complex<double> Cos(const Complex<double>& x) { return std::cos(x); }

inline Complex<long double> Cos(const Complex<long double>& x) {
  return std::cos(x);
}

template <typename Real>
Complex<Real> Cos(const Complex<Real>& x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  return Complex<Real>(real_cos * imag_cosh, -real_sin * imag_sinh);
}

inline void SinCos(const Complex<float>& x, Complex<float>* sin_x,
                   Complex<float>* cos_x) {
  *sin_x = std::sin(x);
  *cos_x = std::cos(x);
}

inline void SinCos(const Complex<double>& x, Complex<double>* sin_x,
                   Complex<double>* cos_x) {
  *sin_x = std::sin(x);
  *cos_x = std::cos(x);
}

inline void SinCos(const Complex<long double>& x, Complex<long double>* sin_x,
                   Complex<long double>* cos_x) {
  *sin_x = std::sin(x);
  *cos_x = std::cos(x);
}

template <typename Real>
void SinCos(const Complex<Real>& x, Complex<Real>* sin_x,
            Complex<Real>* cos_x) {
  Real real_sin, real_cos;
  SinCos(x.Real(), &real_sin, &real_cos);
  Real imag_sinh, imag_cosh;
  HyperbolicSinCos(x.Imag(), &imag_sinh, &imag_cosh);
  *sin_x = Complex<Real>(real_sin * imag_cosh, real_cos * imag_sinh);
  *cos_x = Complex<Real>(real_cos * imag_cosh, -real_sin * imag_sinh);
}

inline Complex<float> Tan(const Complex<float>& x) { return std::tan(x); }

inline Complex<double> Tan(const Complex<double>& x) { return std::tan(x); }

inline Complex<long double> Tan(const Complex<long double>& x) {
  return std::tan(x);
}

template <typename Real>
Complex<Real> Tan(const Complex<Real>& x) {
  Complex<Real> sin_x, cos_x;
  SinCos(x, &sin_x, &cos_x);
  return sin_x / cos_x;
}

inline Complex<float> ArcHyperbolicSin(const Complex<float>& x) {
  return std::asinh(x);
}

inline Complex<double> ArcHyperbolicSin(const Complex<double>& x) {
  return std::asinh(x);
}

inline Complex<long double> ArcHyperbolicSin(const Complex<long double>& x) {
  return std::asinh(x);
}

template <typename Real>
Complex<Real> ArcHyperbolicSin(const Complex<Real>& x) {
  // TODO(Jack Poulson): Document the formula.
  Complex<Real> tau(Real(1) + (x.Real() - x.Imag()) * (x.Real() + x.Imag()),
                    Real(2) * x.Real() * x.Imag());
  tau = SquareRoot(tau);
  return Log(tau + x);
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const Complex<Real>& value) {
  out << RealPart(value) << " + " << ImagPart(value) << "i";
  return out;
}

}  // namespace mantis

#endif  // ifndef MANTIS_COMPLEX_IMPL_H_
