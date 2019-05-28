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
  const int num_bits =
      std::numeric_limits<mantis::DoubleMantissa<Real>>::digits;
  const mantis::DoubleMantissa<Real> epsilon =
      std::numeric_limits<mantis::DoubleMantissa<Real>>::epsilon();
  std::cout << "num bits: " << num_bits << ", epsilon: " << epsilon
            << std::endl;

  const mantis::DoubleMantissa<Real> x("1.2345678901234567890123456789012e1");
  const mantis::DecimalNotation y_decimal{true, 1, std::vector<unsigned char>{
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
  const mantis::DoubleMantissa<Real> y(y_decimal);
  const mantis::DoubleMantissa<Real> z = x - y;
  std::cout << "x: " << x << ",\ny: " << y << ",\nx - y: " << z << std::endl;

  const mantis::DoubleMantissa<Real> x_exp = std::exp(x);
  const mantis::DoubleMantissa<Real> x_exp_log = std::log(x_exp);
  const mantis::DoubleMantissa<Real> x_exp_log_error = x - x_exp_log; 
  std::cout << "exp(x): " << x_exp << ",\nlog(exp(x)): "
            << x_exp_log << ",\nx - log(exp(x)): " << x_exp_log_error
            << std::endl;

  const mantis::BinaryNotation x_binary = x.ToBinary(num_bits);
  std::cout << "x binary: " << x_binary.ToString() << std::endl;

  std::mt19937 generator(17u);
  std::uniform_real_distribution<mantis::DoubleMantissa<Real>> uniform_dist;
  const int num_samples = 1000000;
  mantis::DoubleMantissa<Real> average;
  for (int sample = 0; sample < num_samples; ++sample) {
    average += uniform_dist(generator) / Real(num_samples);
  }
  std::cout << "Average of " << num_samples << " samples: " << average
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Testing with DoubleMantissa<float>:" << std::endl;
  RunTest<float>();
  std::cout << std::endl;

  std::cout << "Testing with DoubleMantissa<double>:" << std::endl;
  RunTest<double>();

  return 0;
}
