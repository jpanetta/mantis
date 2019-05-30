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

  std::cout << "a: " << a << "\n"
            << "b: " << b << "\n"
            << "c = a * b: " << c << "\n"
            << "d = c / a = b: " << d << "\n"
            << "e = c / b = a: " << e << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Testing with DoubleMantissa<float>:" << std::endl;
  RunTest<float>();
  std::cout << std::endl;

  std::cout << "Testing with DoubleMantissa<double>:" << std::endl;
  RunTest<double>();

  return 0;
}
