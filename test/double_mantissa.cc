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

TEST_CASE("Add [float]", "[Add float]") {
  DoubleMantissa<float> value(1.f, 2.f);
  REQUIRE(value.Upper() == 1.f);
  REQUIRE(value.Lower() == 2.f);

  value += 5.f;

  // double can exactly represent up to 24-bit integers.
  REQUIRE(value.Upper() == 8.f);
  REQUIRE(value.Lower() == 0.f);

  value += DoubleMantissa<float>(3.f, 4.f);

  // double can exactly represent up to 24-bit integers.
  REQUIRE(value.Upper() == 15.f);
  REQUIRE(value.Lower() == 0.f);
}

TEST_CASE("Add [double]", "[Add double]") {
  mantis::FPUFix fpu_fix;

  DoubleMantissa<double> value(1., 2.);
  REQUIRE(value.Upper() == 1.);
  REQUIRE(value.Lower() == 2.);

  value += 5.;

  // double can exactly represent up to 53-bit integers.
  REQUIRE(value.Upper() == 8.);
  REQUIRE(value.Lower() == 0.);

  value += DoubleMantissa<double>(3., 4.);

  // double can exactly represent up to 53-bit integers.
  REQUIRE(value.Upper() == 15.);
  REQUIRE(value.Lower() == 0.);
}

TEST_CASE("Add [long double]", "[Add long double]") {
  mantis::FPUFix fpu_fix;

  DoubleMantissa<long double> value(1.L, 2.L);
  REQUIRE(value.Upper() == 1.L);
  REQUIRE(value.Lower() == 2.L);

  value += 5.L;

  // long double can exactly represent up to ?-bit integers.
  REQUIRE(value.Upper() == 8.L);
  REQUIRE(value.Lower() == 0.L);

  value += DoubleMantissa<long double>(3.L, 4.L);

  // double can exactly represent up to 53-bit integers.
  REQUIRE(value.Upper() == 15.L);
  REQUIRE(value.Lower() == 0.L);
}

TEST_CASE("Subtract [float]", "[Subtract float]") {
  DoubleMantissa<float> value(1., 2.f);
  REQUIRE(value.Upper() == 1.f);
  REQUIRE(value.Lower() == 2.f);

  value -= -5.f;

  // double can exactly represent up to 24-bit integers.
  REQUIRE(value.Upper() == 8.f);
  REQUIRE(value.Lower() == 0.f);

  value -= DoubleMantissa<float>(-3.f, -4.f);

  // double can exactly represent up to 24-bit integers.
  REQUIRE(value.Upper() == 15.f);
  REQUIRE(value.Lower() == 0.f);
}

TEST_CASE("Subtract [double]", "[Subtract double]") {
  mantis::FPUFix fpu_fix;

  DoubleMantissa<double> value(1., 2.);
  REQUIRE(value.Upper() == 1.);
  REQUIRE(value.Lower() == 2.);

  value -= -5.;

  // double can exactly represent up to 53-bit integers.
  REQUIRE(value.Upper() == 8.);
  REQUIRE(value.Lower() == 0.);

  value -= DoubleMantissa<double>(-3., -4.);

  // double can exactly represent up to 53-bit integers.
  REQUIRE(value.Upper() == 15.);
  REQUIRE(value.Lower() == 0.);
}

TEST_CASE("Multiply [float]", "[Multiply float]") {
  const float eps = std::numeric_limits<float>::epsilon();
  const DoubleMantissa<float> x(3.f, 4e-11f), y(4.f, 5e-11f);
  const DoubleMantissa<float> z = x * y;

  const DoubleMantissa<float> z_expected(12.f, 3.1e-10f);
  const DoubleMantissa<float> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));

  const DoubleMantissa<float> w = 5.f * x;
  const DoubleMantissa<float> w_expected(15.f, 2e-10f);
  const DoubleMantissa<float> w_error = w - w_expected;
  REQUIRE(std::abs(w_error.Upper()) <= 2 * eps * std::abs(w_expected.Upper()));
  REQUIRE(std::abs(w_error.Lower()) <= 2 * eps * std::abs(w_expected.Lower()));
}

TEST_CASE("Multiply [double]", "[Multiply double]") {
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

TEST_CASE("Square [float]", "[Square float]") {
  const float eps = std::numeric_limits<float>::epsilon();
  const DoubleMantissa<float> x(-3.f, 4e-11f);
  const DoubleMantissa<float> z = mantis::Square(x);

  const DoubleMantissa<float> z_expected(9.f, -2.4e-10f);
  const DoubleMantissa<float> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));
}

TEST_CASE("Square [double]", "[Square double]") {
  mantis::FPUFix fpu_fix;

  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(-3., 4e-20);
  const DoubleMantissa<double> z = mantis::Square(x);

  const DoubleMantissa<double> z_expected(9., -2.4e-19);
  const DoubleMantissa<double> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));
}

TEST_CASE("Divide [float]", "[Divide float]") {
  const float eps = std::numeric_limits<float>::epsilon();
  const DoubleMantissa<float> x(3., 4e-11f), y(4., 5e-11f);
  const DoubleMantissa<float> z = x / y;
  const DoubleMantissa<float> z_fast = DoubleMantissa<float>::FastDivide(x, y);

  const DoubleMantissa<float> r = x - y * z;
  REQUIRE(std::abs(r.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(r.Lower()) <= 2 * eps * std::abs(x.Lower()));

  const DoubleMantissa<float> r_fast = x - y * z_fast;
  REQUIRE(std::abs(r_fast.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(r_fast.Lower()) <= 2 * eps * std::abs(x.Lower()));

  const DoubleMantissa<float> w = x / 5.7f;
  const DoubleMantissa<float> u = x - 5.7f * w;
  REQUIRE(std::abs(u.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(u.Lower()) <= 2 * eps * std::abs(x.Lower()));
}

TEST_CASE("Divide [double]", "[Divide double]") {
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

TEST_CASE("Sqrt [float]", "[Sqrt float]") {
  const DoubleMantissa<float> eps =
      std::numeric_limits<DoubleMantissa<float>>::epsilon();

  const std::vector<DoubleMantissa<float>> inputs{
      DoubleMantissa<float>(3.f, 4e-11f).Reduce(),
      DoubleMantissa<float>(4.f, 5e-11f).Reduce(), DoubleMantissa<float>(),
  };

  for (const DoubleMantissa<float>& x : inputs) {
    const DoubleMantissa<float> x_sqrt = std::sqrt(x);
    const DoubleMantissa<float> x_error = x - mantis::Square(x_sqrt);
    REQUIRE(std::abs(x_error.Upper()) <= 3 * eps.Upper() * std::abs(x.Upper()));
  }

  const DoubleMantissa<float> neg_sqrt =
      std::sqrt(DoubleMantissa<float>(-1., 0.));
  REQUIRE(neg_sqrt.Upper() != neg_sqrt.Upper());
}

TEST_CASE("Sqrt [double]", "[Sqrt double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();

  const std::vector<DoubleMantissa<double>> inputs{
      DoubleMantissa<double>(3., 4e-20).Reduce(),
      DoubleMantissa<double>(4., 5e-20).Reduce(), DoubleMantissa<double>(),
  };

  for (const DoubleMantissa<double>& x : inputs) {
    const DoubleMantissa<double> x_sqrt = std::sqrt(x);
    const DoubleMantissa<double> x_error = x - mantis::Square(x_sqrt);
    const double tolerance = 3 * eps.Upper() * std::abs(x.Upper());
    REQUIRE(std::abs(x_error.Upper()) <= tolerance);
  }

  const DoubleMantissa<double> neg_sqrt =
      std::sqrt(DoubleMantissa<double>(-1., 0.));
  REQUIRE(neg_sqrt.Upper() != neg_sqrt.Upper());
}

TEST_CASE("Exp [float]", "[Exp float]") {
  const DoubleMantissa<float> eps =
      std::numeric_limits<DoubleMantissa<float>>::epsilon();

  {
    const DoubleMantissa<float> log_of_2 =
        mantis::double_mantissa::LogOf2<float>();
    const DoubleMantissa<float> exp_of_log_of_2 = std::exp(log_of_2);
    const DoubleMantissa<float> exp_of_log_of_2_error =
        DoubleMantissa<float>(2) - exp_of_log_of_2;
    REQUIRE(std::abs(exp_of_log_of_2_error.Upper()) <=
            2 * eps.Upper() * float(2));
    REQUIRE(std::abs(exp_of_log_of_2_error.Lower()) <=
            2 * eps.Upper() * float(2));
  }

  const float log_max = 82.f;

  const std::vector<DoubleMantissa<float>> inputs{
      DoubleMantissa<float>(3.f, 4e-11f).Reduce(),
      DoubleMantissa<float>(0.f),
      DoubleMantissa<float>(1.f),
      DoubleMantissa<float>(1e-8f, 1e-19f).Reduce(),
      DoubleMantissa<float>(23.f, 2.5e-9f).Reduce(),
      DoubleMantissa<float>(42.f, 1e-9f).Reduce(),
      DoubleMantissa<float>(100.f, 1e-8f).Reduce(),
      DoubleMantissa<float>(1e5f),
      DoubleMantissa<float>(1e6f, 1e-3f).Reduce(),
  };

  for (const DoubleMantissa<float>& x : inputs) {
    if (x.Upper() > 0) {
      const DoubleMantissa<float> x_log = std::log(x);
      const DoubleMantissa<float> x_log_exp = std::exp(x_log);
      const DoubleMantissa<float> x_log_exp_error = x - x_log_exp;
      const float upper_error = std::abs(x_log_exp_error.Upper());

      const float tolerance =
          3 * eps.Upper() * std::max(1.f, std::abs(x.Upper()));
      REQUIRE(upper_error <= tolerance);
    }

    if (x.Upper() < log_max) {
      const DoubleMantissa<float> x_exp = std::exp(x);
      const DoubleMantissa<float> x_exp_log = std::log(x_exp);
      const DoubleMantissa<float> x_exp_log_error = x - x_exp_log;
      const float upper_error = std::abs(x_exp_log_error.Upper());
      const float tolerance =
          3 * eps.Upper() * std::max(1.f, std::abs(x.Upper()));
      REQUIRE(upper_error <= tolerance);
    }
  }
}

TEST_CASE("Exp [double]", "[Exp double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();

  {
    const DoubleMantissa<double> log_of_2 =
        mantis::double_mantissa::LogOf2<double>();
    const DoubleMantissa<double> exp_of_log_of_2 = std::exp(log_of_2);
    const DoubleMantissa<double> exp_of_log_of_2_error =
        DoubleMantissa<double>(2) - exp_of_log_of_2;
    REQUIRE(std::abs(exp_of_log_of_2_error.Upper()) <=
            2 * eps.Upper() * double(2));
  }

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
          3 * eps.Upper() * std::max(1., std::abs(x.Upper()));
      REQUIRE(upper_error <= tolerance);
    }

    const DoubleMantissa<double> x_exp = std::exp(x);
    const DoubleMantissa<double> x_exp_log = std::log(x_exp);
    const DoubleMantissa<double> x_exp_log_error = x - x_exp_log;
    const double upper_error = std::abs(x_exp_log_error.Upper());
    const double tolerance =
        3 * eps.Upper() * std::max(1., std::abs(x.Upper()));
    REQUIRE(upper_error <= tolerance);
  }
}

TEST_CASE("Log10 [float]", "[Log10 float]") {
  const DoubleMantissa<float> eps =
      std::numeric_limits<DoubleMantissa<float>>::epsilon();

  {
    const DoubleMantissa<float> log_of_10 =
        mantis::double_mantissa::LogOf10<float>();
    const DoubleMantissa<float> exp_of_log_of_10 = std::exp(log_of_10);
    const DoubleMantissa<float> exp_of_log_of_10_error =
        DoubleMantissa<float>(10) - exp_of_log_of_10;
    const float error = std::abs(exp_of_log_of_10_error.Upper());
    const float tolerance = 2 * eps.Upper() * 10;
    REQUIRE(error <= tolerance);
  }

  const std::vector<std::pair<DoubleMantissa<float>, DoubleMantissa<float>>>
      tests{
          // We purposely pick a value that is representable exactly as a
          // float, as the significand is 24-bits.
          std::make_pair(DoubleMantissa<float>(1.e5f),
                         DoubleMantissa<float>(5.f)),
          std::make_pair(DoubleMantissa<float>(1.e2f),
                         DoubleMantissa<float>(2.f)),
          std::make_pair(
              DoubleMantissa<float>(0),
              std::numeric_limits<DoubleMantissa<float>>::quiet_NaN()),
          std::make_pair(
              DoubleMantissa<float>(-1),
              std::numeric_limits<DoubleMantissa<float>>::quiet_NaN()),
      };

  for (const auto& pair : tests) {
    const DoubleMantissa<float>& x = pair.first;
    const DoubleMantissa<float>& x_log10_expected = pair.second;

    const DoubleMantissa<float> x_log10 = std::log10(x);
    const DoubleMantissa<float> x_log10_error = x_log10_expected - x_log10;
    if (x.Upper() > 0) {
      const float upper_error = std::abs(x_log10_error.Upper());
      const float tolerance =
          3 * eps.Upper() * std::max(1.f, std::abs(x_log10_expected.Upper()));
      REQUIRE(upper_error <= tolerance);
    } else {
      REQUIRE(x_log10 != x_log10);
    }
  }
}

TEST_CASE("Log10 [double]", "[Log10 double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();

  {
    const DoubleMantissa<double> log_of_10 =
        mantis::double_mantissa::LogOf10<double>();
    const DoubleMantissa<double> exp_of_log_of_10 = std::exp(log_of_10);
    const DoubleMantissa<double> exp_of_log_of_10_error =
        DoubleMantissa<double>(10) - exp_of_log_of_10;
    const double error = std::abs(exp_of_log_of_10_error.Upper());
    const double tolerance = 2 * eps.Upper() * 10;
    REQUIRE(error <= tolerance);
  }

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
          3 * eps.Upper() * std::max(1., std::abs(x_log10_expected.Upper()));
      REQUIRE(upper_error <= tolerance);
    } else {
      REQUIRE(x_log10 != x_log10);
    }
  }
}

TEST_CASE("Round [float]", "[Round float]") {
  const DoubleMantissa<float> eps =
      std::numeric_limits<DoubleMantissa<float>>::epsilon();
  const std::vector<std::pair<DoubleMantissa<float>, DoubleMantissa<float>>>
      tests{
          std::make_pair(DoubleMantissa<float>(1.234e8f),
                         DoubleMantissa<float>(1.234e8f)),
          std::make_pair(DoubleMantissa<float>(101.34f),
                         DoubleMantissa<float>(101.f)),
          std::make_pair(DoubleMantissa<float>(-2.3f),
                         DoubleMantissa<float>(-2.f)),
          std::make_pair(DoubleMantissa<float>(2.3f),
                         DoubleMantissa<float>(2.f)),
          std::make_pair(DoubleMantissa<float>(2.5f),
                         DoubleMantissa<float>(3.f)),
          std::make_pair(DoubleMantissa<float>(2.4999f),
                         DoubleMantissa<float>(2.f)),
          std::make_pair(DoubleMantissa<float>(-2.5f),
                         DoubleMantissa<float>(-3.f)),
          std::make_pair(DoubleMantissa<float>(-2.4999f),
                         DoubleMantissa<float>(-2.f)),
          std::make_pair(DoubleMantissa<float>(2.f, -1e-20f),
                         DoubleMantissa<float>(2.f)),
          std::make_pair(DoubleMantissa<float>(75689432.5f, 1e-5f),
                         DoubleMantissa<float>(75689433.f)),
          std::make_pair(DoubleMantissa<float>(75689432.5f),
                         DoubleMantissa<float>(75689433.f)),
          std::make_pair(DoubleMantissa<float>(75689432.5f, -1e-5f),
                         DoubleMantissa<float>(75689432.f)),
          std::make_pair(DoubleMantissa<float>(-75689432.5f, 1e-5f),
                         DoubleMantissa<float>(-75689432.f)),
          std::make_pair(DoubleMantissa<float>(-75689432.5f),
                         DoubleMantissa<float>(-75689433.f)),
          std::make_pair(DoubleMantissa<float>(-75689432.5f, -1e-5f),
                         DoubleMantissa<float>(-75689433.f)),
      };

  for (const auto& pair : tests) {
    const DoubleMantissa<float>& x = pair.first;
    const DoubleMantissa<float>& x_round_expected = pair.second;

    const DoubleMantissa<float> x_round = std::round(x);
    const DoubleMantissa<float> x_round_error = x_round_expected - x_round;
    const float upper_error = std::abs(x_round_error.Upper());
    const float tolerance =
        eps.Upper() * std::max(1.f, std::abs(x_round_expected.Upper()));
    REQUIRE(upper_error <= tolerance);
  }
}

TEST_CASE("Round [double]", "[Round double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();
  const std::vector<std::pair<DoubleMantissa<double>, DoubleMantissa<double>>>
      tests{
          std::make_pair(DoubleMantissa<double>(1.234e8),
                         DoubleMantissa<double>(1.234e8)),
          std::make_pair(DoubleMantissa<double>(101.34),
                         DoubleMantissa<double>(101.)),
          std::make_pair(DoubleMantissa<double>(-2.3),
                         DoubleMantissa<double>(-2.)),
          std::make_pair(DoubleMantissa<double>(2.3),
                         DoubleMantissa<double>(2.)),
          std::make_pair(DoubleMantissa<double>(2.5),
                         DoubleMantissa<double>(3.)),
          std::make_pair(DoubleMantissa<double>(2.4999),
                         DoubleMantissa<double>(2.)),
          std::make_pair(DoubleMantissa<double>(-2.5),
                         DoubleMantissa<double>(-3.)),
          std::make_pair(DoubleMantissa<double>(-2.4999),
                         DoubleMantissa<double>(-2.)),
          std::make_pair(DoubleMantissa<double>(2., -1e-20),
                         DoubleMantissa<double>(2.)),
          std::make_pair(DoubleMantissa<double>(75689432.5, 1e-5),
                         DoubleMantissa<double>(75689433.)),
          std::make_pair(DoubleMantissa<double>(75689432.5),
                         DoubleMantissa<double>(75689433.)),
          std::make_pair(DoubleMantissa<double>(75689432.5, -1e-5),
                         DoubleMantissa<double>(75689432.)),
          std::make_pair(DoubleMantissa<double>(-75689432.5, 1e-5),
                         DoubleMantissa<double>(-75689432.)),
          std::make_pair(DoubleMantissa<double>(-75689432.5),
                         DoubleMantissa<double>(-75689433.)),
          std::make_pair(DoubleMantissa<double>(-75689432.5, -1e-5),
                         DoubleMantissa<double>(-75689433.)),
      };

  for (const auto& pair : tests) {
    const DoubleMantissa<double>& x = pair.first;
    const DoubleMantissa<double>& x_round_expected = pair.second;

    const DoubleMantissa<double> x_round = std::round(x);
    const DoubleMantissa<double> x_round_error = x_round_expected - x_round;
    const double upper_error = std::abs(x_round_error.Upper());
    const double tolerance =
        eps.Upper() * std::max(1., std::abs(x_round_expected.Upper()));
    REQUIRE(upper_error <= tolerance);
  }
}

TEST_CASE("Abs [double]", "[Abs double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();
  const std::vector<std::pair<DoubleMantissa<double>, DoubleMantissa<double>>>
      tests{
          std::make_pair(DoubleMantissa<double>(1.234e8),
                         DoubleMantissa<double>(1.234e8)),
          std::make_pair(DoubleMantissa<double>(101.34),
                         DoubleMantissa<double>(101.34)),
          std::make_pair(DoubleMantissa<double>(-2.3),
                         DoubleMantissa<double>(2.3)),
          std::make_pair(DoubleMantissa<double>(2.3),
                         DoubleMantissa<double>(2.3)),
          std::make_pair(DoubleMantissa<double>(2., -1e-20),
                         DoubleMantissa<double>(2., -1e-20)),
          std::make_pair(DoubleMantissa<double>(75689432.5, 1e-5),
                         DoubleMantissa<double>(75689432.5, 1e-5)),
          std::make_pair(DoubleMantissa<double>(75689432.5, -1e-5),
                         DoubleMantissa<double>(75689432.5, -1e-5)),
          std::make_pair(DoubleMantissa<double>(-75689432.5, 1e-5),
                         DoubleMantissa<double>(75689432.5, -1e-5)),
          std::make_pair(DoubleMantissa<double>(-75689432.5, -1e-5),
                         DoubleMantissa<double>(75689432.5, 1e-5)),
      };

  for (const auto& pair : tests) {
    const DoubleMantissa<double>& x = pair.first;
    const DoubleMantissa<double>& x_abs_expected = pair.second;

    const DoubleMantissa<double> x_abs = std::abs(x);
    const DoubleMantissa<double> x_abs_error = x_abs_expected - x_abs;
    const double upper_error = std::abs(x_abs_error.Upper());
    const double tolerance =
        eps.Upper() * std::max(1., std::abs(x_abs_expected.Upper()));
    REQUIRE(upper_error <= tolerance);
  }
}

TEST_CASE("SinCos [double]", "[SinCos double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();
  const DoubleMantissa<double> pi = mantis::double_mantissa::Pi<double>();
  const DoubleMantissa<double> two_pi = 2. * pi;
  const DoubleMantissa<double> half_pi = 0.5 * pi;

  const std::vector<DoubleMantissa<double>> inputs{
      DoubleMantissa<double>(0.),       DoubleMantissa<double>(1.) / 100.,
      DoubleMantissa<double>(1.) / 50., DoubleMantissa<double>(1.) / 25.,
      DoubleMantissa<double>(0.06),     DoubleMantissa<double>(1.) / 10.,
      DoubleMantissa<double>(1.) / 5.,  DoubleMantissa<double>(0.3),
      DoubleMantissa<double>(0.4),      DoubleMantissa<double>(0.5),
      DoubleMantissa<double>(1.),       DoubleMantissa<double>(3.14159),
      DoubleMantissa<double>(-3.14159), DoubleMantissa<double>(6.28318),
  };

  for (const auto& theta : inputs) {
    const DoubleMantissa<double> sin_theta = std::sin(theta);
    const DoubleMantissa<double> cos_theta = std::cos(theta);
    const DoubleMantissa<double> error =
        DoubleMantissa<double>(1.) - Square(sin_theta) - Square(cos_theta);
    REQUIRE(std::abs(error.Upper()) <= 2. * eps.Upper());

    // Push theta into (-pi, pi].
    DoubleMantissa<double> theta_reduced = theta;
    while (theta_reduced > pi) {
      theta_reduced -= two_pi;
    }
    while (theta_reduced <= -pi) {
      theta_reduced += two_pi;
    }

    // Push theta_sin_reduced into [-pi/2, pi/2].
    DoubleMantissa<double> theta_sin_reduced = theta_reduced;
    if (theta_reduced > half_pi) {
      theta_sin_reduced = pi - theta_reduced;
    }
    if (theta_reduced < -half_pi) {
      theta_sin_reduced = -pi - theta_reduced;
    }

    // Push theta_cos_reduced into [0, pi].
    const DoubleMantissa<double> theta_cos_reduced = std::abs(theta_reduced);

    // TODO(Jack Poulson): Come up with a more principled error tolerance. The
    // error grows substantially near the boundaries.
    const DoubleMantissa<double> arc_sin_sin_theta = std::asin(sin_theta);
    const double arc_sin_sin_theta_double = std::asin(std::sin(theta.Upper()));
    const double arc_sin_sin_theta_double_error =
        std::abs(arc_sin_sin_theta_double - theta_sin_reduced);
    const DoubleMantissa<double> arc_sin_sin_theta_error =
        theta_sin_reduced - arc_sin_sin_theta;
    const double arc_sin_sin_theta_tol = std::max(
        4. * eps.Upper(), 2. * std::pow(arc_sin_sin_theta_double_error, 2.));
    REQUIRE(std::abs(arc_sin_sin_theta_error.Upper()) <= arc_sin_sin_theta_tol);

    // TODO(Jack Poulson): Come up with a more principled error tolerance. The
    // error grows substantially near the boundaries.
    const DoubleMantissa<double> arc_cos_cos_theta = std::acos(cos_theta);
    const double arc_cos_cos_theta_double = std::acos(std::cos(theta.Upper()));
    const DoubleMantissa<double> arc_cos_cos_theta_error =
        theta_cos_reduced - arc_cos_cos_theta;
    const double arc_cos_cos_theta_double_error =
        std::abs(arc_cos_cos_theta_double - theta_cos_reduced.Upper());
    const double arc_cos_cos_theta_tol = std::max(
        4. * eps.Upper(), 2. * std::pow(arc_cos_cos_theta_double_error, 2.));
    REQUIRE(std::abs(arc_cos_cos_theta_error.Upper()) <= arc_cos_cos_theta_tol);

    // TODO(Jack Poulson): Come up with a more principled error tolerance. The
    // error grows substantially near the boundaries.
    const DoubleMantissa<double> arc_tan_sin_cos_theta =
        std::atan2(sin_theta, cos_theta);
    const DoubleMantissa<double> arc_tan_sin_cos_theta_error =
        theta_sin_reduced - arc_tan_sin_cos_theta;
    const double arc_tan_sin_cos_theta_double =
        std::atan2(std::sin(theta), std::cos(theta));
    const double arc_tan_sin_cos_theta_double_error =
        std::abs(arc_tan_sin_cos_theta_double - theta_sin_reduced.Upper());
    const double arc_tan_sin_cos_theta_tol =
        std::max(4. * eps.Upper(),
                 2. * std::pow(arc_tan_sin_cos_theta_double_error, 2.));
    REQUIRE(std::abs(arc_tan_sin_cos_theta_error.Upper()) <=
            arc_tan_sin_cos_theta_tol);
  }
}
