![](./images/mantis-50pct.png)

[**mantis**](https://gitlab.com/hodge_star/mantis) is a
[C++14](https://en.wikipedia.org/wiki/C%2B%2B14), header-only implementation of
real and complex double-mantissa arithmetic.

The library is meant as a revival of the [QD library](https://github.com/scibuilder/QD) of Yozo Hida, Xiaoye S. Li, and David H. Bailey, which is described in:

    Yozo Hida, Xiaoye S. Li, and David H. Bailey,
    "Library for Double-Double and Quad-Double Arithmetic", May 8, 2008.
    URL: [www.davidhbailey.com/dhbpapers/qd.pdf](https://www.davidhbailey.com/dhbpapers/qd.pdf)

The original library does not appear to have an official distribution channel,
but it is unofficially available at [github.com/scibuilder/QD](https://github.com/scibuilder/QD).

[![Join the chat at https://gitter.im/hodge_star/community](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/hodge_star/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

### Dependencies
There are no dependencies beyond C++14 for manually including mantis's headers
into your project.

But, if you would like to make use of the project's build system, the only
strict dependency is:

* [meson](http://mesonbuild.com): "Meson is an open source build system meant
to be both extremely fast, and, even more importantly, as user friendly as
possible."

Meson will automatically install
[Catch2](https://github.com/catchorg/Catch2) (a header-only C++
unit-testing library) and [specify](https://gitlab.com/hodge_star/specify)
(a C++14 header-only, command-line argument processor).

### Example usage

Consider the [basic example driver](https://gitlab.com/hodge_star/mantis/blob/master/example/basic.cc):
```c++
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
      1_uchar, 2_uchar, 3_uchar, 4_uchar, 5_uchar, 6_uchar, 7_uchar, 8_uchar,
      9_uchar, 0_uchar, 1_uchar, 2_uchar, 3_uchar, 4_uchar, 5_uchar, 6_uchar,
      7_uchar, 8_uchar, 9_uchar, 0_uchar, 1_uchar, 2_uchar, 3_uchar, 4_uchar,
      5_uchar, 6_uchar, 7_uchar, 8_uchar, 9_uchar, 0_uchar, 1_uchar, 2_uchar}};
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
  std::cout << "Average of " << num_samples << " uniform samples: " << average
            << std::endl;

  std::normal_distribution<mantis::DoubleMantissa<Real>> normal_dist;
  average = Real(0.);
  for (int sample = 0; sample < num_samples; ++sample) {
    average += normal_dist(generator) / Real(num_samples);
  }
  std::cout << "Average of " << num_samples << " normal samples: " << average
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
```

Running this driver might produce:
```
Testing with DoubleMantissa<float>:
num bits: 48, epsilon: 7.105427357601002e-15
x: 1.234567890123458e1,
y: 1.234567890123458e1,
x - y: 0.000000000000000
exp(x): 2.299641948529960e5,
log(exp(x)): 1.234567890123458e1,
x - log(exp(x)): 0.000000000000000
x binary: 0.110001011000011111100110100110010111101110000100e4
Average of 1000000 uniform samples: 4.998531398631090e-1
Average of 1000000 normal samples: -7.498842402897466e-4

Testing with DoubleMantissa<double>:
num bits: 106, epsilon: 2.46519032881566189191165176650871e-32
x: 1.23456789012345678901234567890117e1,
y: 1.23456789012345678901234567890117e1,
x - y: 0.00000000000000000000000000000000
exp(x): 2.29964194852988545212647771928755e5,
log(exp(x)): 1.23456789012345678901234567890117e1,
x - log(exp(x)): 0.00000000000000000000000000000000
x binary: 0.1100010110000111111001101001100101111011100000111101001001110000100011010110110011100011001000110010001111e4
Average of 1000000 uniform samples: 4.99685256909933771504553286824881e-1
Average of 1000000 normal samples: -1.40344112202639366376674728290691e-3
```

### License
`mantis` is distributed under the
[Mozilla Public License, v. 2.0](https://www.mozilla.org/media/MPL/2.0/index.815ca599c9df.txt).
