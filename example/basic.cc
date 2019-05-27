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
  const mantis::DoubleMantissa<Real> x("1.2345678901234567890123456789012e1");
  const mantis::ScientificNotation y_rep{true, 1, std::vector<unsigned char>{
      1_uchar,
      2_uchar,
      3_uchar,
      4_uchar,
      5_uchar,
      6_uchar,
      7_uchar,
      8_uchar,
      9_uchar,
      0_uchar,
      1_uchar,
      2_uchar,
      3_uchar,
      4_uchar,
      5_uchar,
      6_uchar,
      7_uchar,
      8_uchar,
      9_uchar,
      0_uchar,
      1_uchar,
      2_uchar,
      3_uchar,
      4_uchar,
      5_uchar,
      6_uchar,
      7_uchar,
      8_uchar,
      9_uchar,
      0_uchar,
      1_uchar,
      2_uchar}};
  const mantis::DoubleMantissa<Real> y(y_rep);
  const mantis::DoubleMantissa<Real> z = x - y;
  std::cout << "x: " << x << ",\ny: " << y << ",\nx - y: " << z << std::endl;

  const mantis::DoubleMantissa<Real> x_exp = std::exp(x);
  const mantis::DoubleMantissa<Real> x_exp_log = std::log(x_exp);
  const mantis::DoubleMantissa<Real> x_exp_log_error = x - x_exp_log; 
  std::cout << "exp(x): " << x_exp << ",\nlog(exp(x)): "
            << x_exp_log << ",\nx - log(exp(x)): " << x_exp_log_error
            << std::endl;

  const mantis::DoubleMantissa<Real> epsilon =
      std::numeric_limits<mantis::DoubleMantissa<Real>>::epsilon();
  std::cout << "epsilon: " << epsilon << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Testing with DoubleMantissa<float>:" << std::endl;
  RunTest<float>();
  std::cout << std::endl;

  std::cout << "Testing with DoubleMantissa<double>:" << std::endl;
  RunTest<double>();

  return 0;
}
