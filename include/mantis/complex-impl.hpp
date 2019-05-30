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

inline Complex<float> ArcSin(const Complex<float>& x) { return std::asin(x); }

inline Complex<double> ArcSin(const Complex<double>& x) { return std::asin(x); }

inline Complex<long double> ArcSin(const Complex<long double>& x) {
  return std::asin(x);
}

template <typename Real>
Complex<Real> ArcSin(const Complex<Real>& x) {
  // Arcsin(x) = -i Arcsinh(i x)
  const Complex<Real> y = ArcHyperbolicSin(Complex<Real>(-x.Imag(), x.Real()));
  return Complex<Real>(y.Imag(), -y.Real());
}

inline Complex<float> ArcCos(const Complex<float>& x) { return std::acos(x); }

inline Complex<double> ArcCos(const Complex<double>& x) { return std::acos(x); }

inline Complex<long double> ArcCos(const Complex<long double>& x) {
  return std::acos(x);
}

template <typename Real>
Complex<Real> ArcCos(const Complex<Real>& x) {
  static const Real pi = std::acos(Real(-1));
  static const Real half_pi = pi / Real(2);
  const Complex<Real> asin_x = ArcSin(x);
  return Complex<Real>(half_pi - asin_x.Real(), -asin_x.Imag());
}

inline Complex<float> ArcTan(const Complex<float>& x) { return std::atan(x); }

inline Complex<double> ArcTan(const Complex<double>& x) { return std::atan(x); }

inline Complex<long double> ArcTan(const Complex<long double>& x) {
  return std::atan(x);
}

template <typename Real>
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

inline Complex<float> ArcHyperbolicCos(const Complex<float>& x) {
  return std::acosh(x);
}

inline Complex<double> ArcHyperbolicCos(const Complex<double>& x) {
  return std::acosh(x);
}

inline Complex<long double> ArcHyperbolicCos(const Complex<long double>& x) {
  return std::acosh(x);
}

template <typename Real>
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

inline Complex<float> ArcHyperbolicTan(const Complex<float>& x) {
  return std::atanh(x);
}

inline Complex<double> ArcHyperbolicTan(const Complex<double>& x) {
  return std::atanh(x);
}

inline Complex<long double> ArcHyperbolicTan(const Complex<long double>& x) {
  return std::atanh(x);
}

template <typename Real>
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

#endif  // ifndef MANTIS_COMPLEX_IMPL_H_
