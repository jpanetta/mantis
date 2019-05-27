![](./images/mantis-50pct.png)

[**mantis**](https://gitlab.com/hodge_star/mantis) is a
[C++14](https://en.wikipedia.org/wiki/C%2B%2B14), header-only implementation of
double-mantissa arithmetic.

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
  const mantis::DoubleMantissa<Real> x("1.2345678901234567890123456789012e1");
  const mantis::ScientificNotation y_rep{true, 1, std::vector<unsigned char>{
      1_uchar, 2_uchar, 3_uchar, 4_uchar, 5_uchar, 6_uchar, 7_uchar, 8_uchar,
      9_uchar, 0_uchar, 1_uchar, 2_uchar, 3_uchar, 4_uchar, 5_uchar, 6_uchar,
      7_uchar, 8_uchar, 9_uchar, 0_uchar, 1_uchar, 2_uchar, 3_uchar, 4_uchar,
      5_uchar, 6_uchar, 7_uchar, 8_uchar, 9_uchar, 0_uchar, 1_uchar, 2_uchar}};
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
```

Running this driver produces:
```
Testing with DoubleMantissa<float>:
x: 1.234567890123458e1,
y: 1.234567890123458e1,
x - y: 0.000000000000000
exp(x): 2.299641948529960e5,
log(exp(x)): 1.234567890123458e1,
x - log(exp(x)): 0.000000000000000
epsilon: 7.105427357601002e-15

Testing with DoubleMantissa<double>:
x: 1.23456789012345678901234567890117e1,
y: 1.23456789012345678901234567890117e1,
x - y: 0.00000000000000000000000000000000
exp(x): 2.29964194852988545212647771928856e5,
log(exp(x)): 1.23456789012345678901234567890117e1,
x - log(exp(x)): 0.00000000000000000000000000000000
epsilon: 2.46519032881566189191165176650871e-32
```

### License
`mantis` is distributed under the
[Mozilla Public License, v. 2.0](https://www.mozilla.org/media/MPL/2.0/index.815ca599c9df.txt).
