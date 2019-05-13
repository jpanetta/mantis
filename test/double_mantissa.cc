/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <iostream>
#include <limits>
#include "catch2/catch.hpp"
#include "mantis.hpp"
using mantis::DoubleMantissa;

TEST_CASE("Add", "[Add]") {
  mantis::FPUFix fpu_fix;

  DoubleMantissa<double> value(1., 2.);
  REQUIRE(value.Upper() == 1.);
  REQUIRE(value.Lower() == 2.);

  value += 5.;

  // fp64 can exactly represent up to 52-bit integers.
  REQUIRE(value.Upper() == 8.);
  REQUIRE(value.Lower() == 0.);

  value += DoubleMantissa<double>(3., 4.);

  // fp64 can exactly represent up to 52-bit integers.
  REQUIRE(value.Upper() == 15.);
  REQUIRE(value.Lower() == 0.);
}

TEST_CASE("Subtract", "[Subtract]") {
  mantis::FPUFix fpu_fix;

  DoubleMantissa<double> value(1., 2.);
  REQUIRE(value.Upper() == 1.);
  REQUIRE(value.Lower() == 2.);

  value -= -5.;

  // fp64 can exactly represent up to 52-bit integers.
  REQUIRE(value.Upper() == 8.);
  REQUIRE(value.Lower() == 0.);

  value -= DoubleMantissa<double>(-3., -4.);

  // fp64 can exactly represent up to 52-bit integers.
  REQUIRE(value.Upper() == 15.);
  REQUIRE(value.Lower() == 0.);
}

TEST_CASE("Multiply", "[Multiply]") {
  mantis::FPUFix fpu_fix;

  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(3., 4e-20), y(4., 5e-20);
  const DoubleMantissa<double> z = x * y;

  const DoubleMantissa<double> z_expected(12., 3.1e-19);
  const DoubleMantissa<double> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));

  const DoubleMantissa<double> w = 5. * x;
  const DoubleMantissa<double> w_expected(15., 2e-19);
  const DoubleMantissa<double> w_error = w - w_expected;
  REQUIRE(std::abs(w_error.Upper()) <= 2 * eps * std::abs(w_expected.Upper()));
  REQUIRE(std::abs(w_error.Lower()) <= 2 * eps * std::abs(w_expected.Lower()));
}

TEST_CASE("Square", "[Square]") {
  mantis::FPUFix fpu_fix;

  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(-3., 4e-20);
  const DoubleMantissa<double> z = mantis::Square(x);

  const DoubleMantissa<double> z_expected(9., -2.4e-19);
  const DoubleMantissa<double> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));
}

TEST_CASE("Divide", "[Divide]") {
  mantis::FPUFix fpu_fix;

  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(3., 4e-20), y(4., 5e-20);
  const DoubleMantissa<double> z = x / y;
  const DoubleMantissa<double> z_fast =
      DoubleMantissa<double>::FastDivide(x, y);

  const DoubleMantissa<double> r = x - y * z;
  REQUIRE(std::abs(r.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(r.Lower()) <= 2 * eps * std::abs(x.Lower()));

  const DoubleMantissa<double> r_fast = x - y * z_fast;
  REQUIRE(std::abs(r_fast.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(r_fast.Lower()) <= 2 * eps * std::abs(x.Lower()));

  const DoubleMantissa<double> w = x / 5.7;
  const DoubleMantissa<double> u = x - 5.7 * w;
  REQUIRE(std::abs(u.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(u.Lower()) <= 2 * eps * std::abs(x.Lower()));
}

TEST_CASE("Sqrt", "[Sqrt]") {
  mantis::FPUFix fpu_fix;
  const double eps = std::numeric_limits<double>::epsilon();

  const std::vector<DoubleMantissa<double>> inputs{
      DoubleMantissa<double>(3., 4e-20).Reduce(),
      DoubleMantissa<double>(4., 5e-20).Reduce(), DoubleMantissa<double>(),
  };

  for (const DoubleMantissa<double>& x : inputs) {
    const DoubleMantissa<double> x_sqrt = std::sqrt(x);
    const DoubleMantissa<double> x_error = x - mantis::Square(x_sqrt);
    REQUIRE(std::abs(x_error.Upper()) <= 2 * eps * std::abs(x.Upper()));
  }

  const DoubleMantissa<double> neg_sqrt =
      std::sqrt(DoubleMantissa<double>(-1., 0.));
  REQUIRE(neg_sqrt.Upper() != neg_sqrt.Upper());
}

TEST_CASE("Exp", "[Exp]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps = mantis::double_mantissa::Epsilon<double>();

  const std::vector<DoubleMantissa<double>> inputs{
      DoubleMantissa<double>(3., 4e-20).Reduce(),
      DoubleMantissa<double>(0.),
      DoubleMantissa<double>(1.),
      DoubleMantissa<double>(1e-8, 1e-30).Reduce(),
      DoubleMantissa<double>(101., 1e-18).Reduce(),
  };

  for (const DoubleMantissa<double>& x : inputs) {
    if (x.Upper() > 0) {
      const DoubleMantissa<double> x_log = std::log(x);
      const DoubleMantissa<double> x_log_exp = std::exp(x_log);
      const DoubleMantissa<double> x_log_exp_error = x - x_log_exp;
      const double upper_error = std::abs(x_log_exp_error.Upper());
      const double tolerance =
          5 * eps.Upper() * std::max(1., std::abs(x.Upper()));
      REQUIRE(upper_error <= tolerance);
    }

    const DoubleMantissa<double> x_exp = std::exp(x);
    const DoubleMantissa<double> x_exp_log = std::log(x_exp);
    const DoubleMantissa<double> x_exp_log_error = x - x_exp_log;
    const double upper_error = std::abs(x_exp_log_error.Upper());
    const double tolerance =
        5 * eps.Upper() * std::max(1., std::abs(x.Upper()));
    REQUIRE(upper_error <= tolerance);
  }

  const DoubleMantissa<double> log_of_2 =
      mantis::double_mantissa::LogOf2<double>();
  const DoubleMantissa<double> exp_of_log_of_2 = std::exp(log_of_2);
  const DoubleMantissa<double> exp_of_log_of_2_error =
      DoubleMantissa<double>(2) - exp_of_log_of_2;
  REQUIRE(std::abs(exp_of_log_of_2_error.Upper()) <= 4 * eps.Upper());
}

TEST_CASE("Log10", "[Log10]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps = mantis::double_mantissa::Epsilon<double>();

  const std::vector<std::pair<DoubleMantissa<double>, DoubleMantissa<double>>>
      tests{
          // We purposely pick a value that is representable exactly as a
          // double,
          // as the significand is 53-bits.
          std::make_pair(DoubleMantissa<double>(1.e11),
                         DoubleMantissa<double>(11.)),
          std::make_pair(DoubleMantissa<double>(1.e2),
                         DoubleMantissa<double>(2.)),
          std::make_pair(
              DoubleMantissa<double>(0),
              std::numeric_limits<DoubleMantissa<double>>::quiet_NaN()),
          std::make_pair(
              DoubleMantissa<double>(-1),
              std::numeric_limits<DoubleMantissa<double>>::quiet_NaN()),
      };

  for (const auto& pair : tests) {
    const DoubleMantissa<double>& x = pair.first;
    const DoubleMantissa<double>& x_log10_expected = pair.second;

    const DoubleMantissa<double> x_log10 = std::log10(x);
    const DoubleMantissa<double> x_log10_error = x_log10_expected - x_log10;
    if (x.Upper() > 0) {
      const double upper_error = std::abs(x_log10_error.Upper());
      const double tolerance =
          5 * eps.Upper() * std::max(1., std::abs(x_log10_expected.Upper()));
      REQUIRE(upper_error <= tolerance);
    } else {
      REQUIRE(x_log10 != x_log10);
    }
  }

  const DoubleMantissa<double> log_of_10 =
      mantis::double_mantissa::LogOf10<double>();
  const DoubleMantissa<double> exp_of_log_of_10 = std::exp(log_of_10);
  const DoubleMantissa<double> exp_of_log_of_10_error =
      DoubleMantissa<double>(10) - exp_of_log_of_10;
  const double error = std::abs(exp_of_log_of_10_error.Upper());
  const double tolerance = 4 * eps.Upper() * 10;
  REQUIRE(error <= tolerance);
}
