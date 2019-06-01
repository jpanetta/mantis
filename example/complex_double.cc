/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include "mantis.hpp"

template <typename Real>
void RunTest() {
  typedef mantis::DoubleMantissa<Real> DoubleReal;

  const mantis::Complex<DoubleReal> a{DoubleReal(1.), DoubleReal(2.)};
  const mantis::Complex<DoubleReal> b{DoubleReal(-1.), DoubleReal(3.)};
  const mantis::Complex<DoubleReal> c = a * b;
  const mantis::Complex<DoubleReal> d = c / a;
  const mantis::Complex<DoubleReal> e = c / b;

  std::cout << "a: " << a << ", |a|: " << std::abs(a) << "\n"
            << "b: " << b << ", |b|: " << std::abs(b) << "\n"
            << "c = a * b: " << c << ", |c|: " << std::abs(c) << "\n"
            << "d = c / a = b: " << d << ", |d|: " << std::abs(d) << "\n"
            << "e = c / b = a: " << e << ", |e|: " << std::abs(e)
            << std::endl;

  std::cout << "sin(a): " << std::sin(a) << ", cos(a): " << std::cos(a)
            << ", tan(a): " << std::tan(a) << std::endl;
  std::cout << "sin(b): " << std::sin(b) << ", cos(b): " << std::cos(b)
            << ", tan(b): " << std::tan(b) << std::endl;
  std::cout << "sin(c): " << std::sin(c) << ", cos(c): " << std::cos(c)
            << ", tan(c): " << std::tan(c) << std::endl;

  std::cout << "log(a): " << std::log(a) << std::endl;
  std::cout << "sqrt(a): " << std::sqrt(a) << std::endl;
  std::cout << "arg(a): " << std::arg(a) << std::endl;
  std::cout << "asin(a):  " << std::asin(a) << std::endl;
  std::cout << "acos(a):  " << std::acos(a) << std::endl;
  std::cout << "atan(a):  " << std::atan(a) << std::endl;
  std::cout << "asinh(a): " << std::asinh(a) << std::endl;
  std::cout << "acosh(a): " << std::acosh(a) << std::endl;
  std::cout << "atanh(a): " << std::atanh(a) << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Testing with DoubleMantissa<float>:" << std::endl;
  RunTest<float>();
  std::cout << std::endl;

  std::cout << "Testing with DoubleMantissa<double>:" << std::endl;
  RunTest<double>();

  return 0;
}
