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

  std::cout << "a: " << a << ", |a|: " << mantis::Abs(a) << "\n"
            << "b: " << b << ", |b|: " << mantis::Abs(b) << "\n"
            << "c = a * b: " << c << ", |c|: " << mantis::Abs(c) << "\n"
            << "d = c / a = b: " << d << ", |d|: " << mantis::Abs(d) << "\n"
            << "e = c / b = a: " << e << ", |e|: " << mantis::Abs(e)
            << std::endl;

  std::cout << "sin(a): " << mantis::Sin(a) << ", cos(a): " << mantis::Cos(a)
            << ", tan(a): " << mantis::Tan(a) << std::endl;
  std::cout << "sin(b): " << mantis::Sin(b) << ", cos(b): " << mantis::Cos(b)
            << ", tan(b): " << mantis::Tan(b) << std::endl;
  std::cout << "sin(c): " << mantis::Sin(c) << ", cos(c): " << mantis::Cos(c)
            << ", tan(c): " << mantis::Tan(c) << std::endl;

  std::cout << "log(a): " << mantis::Log(a) << std::endl;
  std::cout << "sqrt(a): " << mantis::SquareRoot(a) << std::endl;
  std::cout << "arg(a): " << mantis::Arg(a) << std::endl;
  std::cout << "ainh(a): " << mantis::ArcHyperbolicSin(a) << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Testing with DoubleMantissa<float>:" << std::endl;
  RunTest<float>();
  std::cout << std::endl;

  std::cout << "Testing with DoubleMantissa<double>:" << std::endl;
  RunTest<double>();

  return 0;
}
