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
#include <tuple>
#include "catch2/catch.hpp"
#include "mantis.hpp"
using mantis::DecimalScientificNotation;
using mantis::DoubleMantissa;

constexpr unsigned char operator"" _uchar(
    unsigned long long int value) noexcept {
  return static_cast<unsigned char>(value);
}

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

  for (const auto& test : tests) {
    const DoubleMantissa<float>& x = test.first;
    const DoubleMantissa<float>& x_log10_expected = test.second;

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

  for (const auto& test : tests) {
    const DoubleMantissa<double>& x = test.first;
    const DoubleMantissa<double>& x_log10_expected = test.second;

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

  for (const auto& test : tests) {
    const DoubleMantissa<float>& x = test.first;
    const DoubleMantissa<float>& x_round_expected = test.second;

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

  for (const auto& test : tests) {
    const DoubleMantissa<double>& x = test.first;
    const DoubleMantissa<double>& x_round_expected = test.second;

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

  for (const auto& test : tests) {
    const DoubleMantissa<double>& x = test.first;
    const DoubleMantissa<double>& x_abs_expected = test.second;

    const DoubleMantissa<double> x_abs = std::abs(x);
    const DoubleMantissa<double> x_abs_error = x_abs_expected - x_abs;
    const double upper_error = std::abs(x_abs_error.Upper());
    const double tolerance =
        eps.Upper() * std::max(1., std::abs(x_abs_expected.Upper()));
    REQUIRE(upper_error <= tolerance);
  }
}

TEST_CASE("Trig [double]", "[Trig double]") {
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

    // Test that cos^2(theta) + sin^2(theta) = 1.
    const DoubleMantissa<double> unit_error =
        DoubleMantissa<double>(1.) - Square(sin_theta) - Square(cos_theta);
    REQUIRE(std::abs(unit_error.Upper()) <= 2. * eps.Upper());

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

TEST_CASE("HypTrig [double]", "[HypTrig double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();

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
    const DoubleMantissa<double> sinh_theta = std::sinh(theta);
    const DoubleMantissa<double> cosh_theta = std::cosh(theta);
    const DoubleMantissa<double> exp_theta = std::exp(theta);

    // Test that cosh^2(theta) - sinh^2(theta) = 1.
    const DoubleMantissa<double> unit_error =
        DoubleMantissa<double>(1.) + Square(sinh_theta) - Square(cosh_theta);
    const double unit_tolerance =
        2. * eps.Upper() * std::max(1., std::max(Square(sinh_theta).Upper(),
                                                 Square(cosh_theta).Upper()));
    if (std::abs(unit_error.Upper()) > unit_tolerance) {
      std::cout << "theta: " << theta << ", sinh(theta): " << sinh_theta
                << ", cosh(theta): " << cosh_theta
                << ", unit_error: " << unit_error
                << ", tolerance: " << unit_tolerance << std::endl;
    }
    REQUIRE(std::abs(unit_error.Upper()) <= unit_tolerance);

    // Test that cosh(theta) + sinh(theta) = exp(theta).
    const DoubleMantissa<double> exp_error =
        exp_theta - cosh_theta - sinh_theta;
    const double exp_tolerance =
        2. * eps.Upper() *
        std::max(exp_theta.Upper(), std::max(std::abs(cosh_theta.Upper()),
                                             std::abs(sinh_theta.Upper())));
    REQUIRE(std::abs(exp_error.Upper()) <= exp_tolerance);

    // Test that cosh(theta) - sinh(theta) = exp(-theta).
    const DoubleMantissa<double> inv_exp_error =
        Inverse(exp_theta) - cosh_theta + sinh_theta;
    const double inv_exp_tolerance =
        2. * eps.Upper() * std::max(Inverse(exp_theta).Upper(),
                                    std::max(std::abs(cosh_theta.Upper()),
                                             std::abs(sinh_theta.Upper())));
    REQUIRE(std::abs(inv_exp_error.Upper()) <= inv_exp_tolerance);
  }
}

TEST_CASE("Floor [double]", "[Floor double]") {
  mantis::FPUFix fpu_fix;
  const DoubleMantissa<double> eps =
      std::numeric_limits<DoubleMantissa<double>>::epsilon();

  const std::vector<std::pair<DoubleMantissa<double>, DoubleMantissa<double>>>
      tests{
          std::make_pair(DoubleMantissa<double>(0.),
                         DoubleMantissa<double>(0.)),
          std::make_pair(DoubleMantissa<double>(0.01),
                         DoubleMantissa<double>(0.)),
          std::make_pair(DoubleMantissa<double>(-3.14159),
                         DoubleMantissa<double>(-4.)),
          std::make_pair(DoubleMantissa<double>(12345678901234500., 12.3),
                         DoubleMantissa<double>(12345678901234500., 12.)),
          std::make_pair(DoubleMantissa<double>(3.14159),
                         DoubleMantissa<double>(3.)),
      };

  for (const auto& test : tests) {
    const DoubleMantissa<double>& x = test.first;
    const DoubleMantissa<double>& x_floor_expected = test.second;
    const DoubleMantissa<double> x_floor = std::floor(x);
    const DoubleMantissa<double> x_floor_error = x_floor - x_floor_expected;
    const double tolerance = 2. * eps * std::abs(x_floor_expected.Upper());
    REQUIRE(std::abs(x_floor_error.Upper()) <= tolerance);
  }

  REQUIRE(static_cast<int>(DoubleMantissa<double>(0.01)) == 0);
  REQUIRE(static_cast<int>(DoubleMantissa<double>(-0.1)) == -1);
  REQUIRE(static_cast<int>(DoubleMantissa<double>(10.1)) == 10);

  REQUIRE(static_cast<long long int>(DoubleMantissa<double>(
              12345678901234500., 67.5)) == 12345678901234567LL);
}

TEST_CASE("IntegerPower [double]", "[IntegerPower double]") {
  mantis::FPUFix fpu_fix;

  const std::vector<
      std::tuple<DoubleMantissa<double>, int, DoubleMantissa<double>>>
      tests{
          std::make_tuple(DoubleMantissa<double>(10.), 0,
                          DoubleMantissa<double>(1.)),
          std::make_tuple(DoubleMantissa<double>(10.), 1,
                          DoubleMantissa<double>(10.)),
          std::make_tuple(DoubleMantissa<double>(10.), -1,
                          DoubleMantissa<double>(1.) / 10.),
          std::make_tuple(DoubleMantissa<double>(10.), 3,
                          DoubleMantissa<double>(1e3)),
          std::make_tuple(DoubleMantissa<double>(10.), -3,
                          DoubleMantissa<double>(1.) / 1e3),
          std::make_tuple(DoubleMantissa<double>(10.), 15,
                          DoubleMantissa<double>(1e15)),
          std::make_tuple(DoubleMantissa<double>(10.), -15,
                          DoubleMantissa<double>(1.) / 1e15),
          std::make_tuple(DoubleMantissa<double>(0.), 0,
                          mantis::double_mantissa::QuietNan<double>()),
      };

  const double epsilon = std::numeric_limits<DoubleMantissa<double>>::epsilon();
  for (const auto& test : tests) {
    const DoubleMantissa<double>& value = std::get<0>(test);
    const int exponent = std::get<1>(test);
    const DoubleMantissa<double>& expected_result = std::get<2>(test);
    const DoubleMantissa<double> result = std::pow(value, exponent);

    if (std::isfinite(expected_result)) {
      const DoubleMantissa<double> error = expected_result - result;
      const double tolerance = epsilon * std::abs(expected_result.Upper());
      REQUIRE(std::abs(error.Upper()) <= tolerance);
    } else if (std::isnan(expected_result)) {
      REQUIRE(std::isnan(result));
    } else if (std::isinf(expected_result)) {
      REQUIRE(result == expected_result);
    }
  }
}

TEST_CASE("Power [double]", "[Power double]") {
  mantis::FPUFix fpu_fix;

  const std::vector<std::tuple<DoubleMantissa<double>, DoubleMantissa<double>,
                               DoubleMantissa<double>>>
      tests{
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(0.),
                          DoubleMantissa<double>(1.)),
          std::make_tuple(DoubleMantissa<double>(16.),
                          DoubleMantissa<double>(0.5),
                          DoubleMantissa<double>(4.)),
          std::make_tuple(DoubleMantissa<double>(16.),
                          DoubleMantissa<double>(1.5),
                          DoubleMantissa<double>(64.)),
          std::make_tuple(DoubleMantissa<double>(1000.),
                          DoubleMantissa<double>(1.) / 3.,
                          DoubleMantissa<double>(10.)),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(1.),
                          DoubleMantissa<double>(10.)),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(-1.),
                          DoubleMantissa<double>(1.) / 10.),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(3.),
                          DoubleMantissa<double>(1e3)),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(-3.),
                          DoubleMantissa<double>(1.) / 1e3),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(15.),
                          DoubleMantissa<double>(1e15)),
          std::make_tuple(DoubleMantissa<double>(10.),
                          DoubleMantissa<double>(-15.),
                          DoubleMantissa<double>(1.) / 1e15),
          std::make_tuple(DoubleMantissa<double>(0.),
                          DoubleMantissa<double>(0.),
                          mantis::double_mantissa::QuietNan<double>()),
      };

  const double epsilon = std::numeric_limits<DoubleMantissa<double>>::epsilon();
  for (const auto& test : tests) {
    const DoubleMantissa<double>& value = std::get<0>(test);
    const DoubleMantissa<double>& exponent = std::get<1>(test);
    const DoubleMantissa<double>& expected_result = std::get<2>(test);
    const DoubleMantissa<double> result = std::pow(value, exponent);

    if (std::isfinite(expected_result)) {
      const DoubleMantissa<double> error = expected_result - result;
      const double tolerance =
          1.5 * epsilon * std::abs(expected_result.Upper());
      REQUIRE(std::abs(error.Upper()) <= tolerance);
    } else if (std::isnan(expected_result)) {
      REQUIRE(std::isnan(result));
    } else if (std::isinf(expected_result)) {
      REQUIRE(result == expected_result);
    }
  }
}

TEST_CASE("ScientificNotation [double]", "[ScientificNotation double]") {
  mantis::FPUFix fpu_fix;

  const std::vector<
      std::tuple<DoubleMantissa<double>, int, DecimalScientificNotation>>
      tests{
          std::make_tuple(
              DoubleMantissa<double>(3.14159), 5,
              DecimalScientificNotation{
                  true, 0, std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar,
                                                      1_uchar, 6_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(3.14159), 4,
              DecimalScientificNotation{
                  true, 0, std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar,
                                                      2_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(-3.14159), 5,
              DecimalScientificNotation{
                  false, 0,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             6_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(113.1238), 6,
              DecimalScientificNotation{
                  true, 2,
                  std::vector<unsigned char>{1_uchar, 1_uchar, 3_uchar, 1_uchar,
                                             2_uchar, 4_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(1.131238e49), 6,
              DecimalScientificNotation{
                  true, 49,
                  std::vector<unsigned char>{1_uchar, 1_uchar, 3_uchar, 1_uchar,
                                             2_uchar, 4_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(1.131238e204), 6,
              DecimalScientificNotation{
                  true,
                  204, std::vector<unsigned char>{1_uchar, 1_uchar, 3_uchar,
                                                  1_uchar, 2_uchar, 4_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(0.001131238), 6,
              DecimalScientificNotation{
                  true, -3,
                  std::vector<unsigned char>{1_uchar, 1_uchar, 3_uchar, 1_uchar,
                                             2_uchar, 4_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(1.131238e-300), 6,
              DecimalScientificNotation{
                  true, -300,
                  std::vector<unsigned char>{1_uchar, 1_uchar, 3_uchar, 1_uchar,
                                             2_uchar, 4_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(9.9999999), 5,
              DecimalScientificNotation{
                  true, 1, std::vector<unsigned char>{1_uchar, 0_uchar, 0_uchar,
                                                      0_uchar, 0_uchar}}),
          std::make_tuple(
              DoubleMantissa<double>(), 6,
              DecimalScientificNotation{
                  true, 0,
                  std::vector<unsigned char>{0_uchar, 0_uchar, 0_uchar, 0_uchar,
                                             0_uchar, 0_uchar}}),
          std::make_tuple(
              mantis::double_mantissa::QuietNan<double>(), 8,
              DecimalScientificNotation{
                  true, 0, std::vector<unsigned char>{'n', 'a', 'n'}}),
          std::make_tuple(
              mantis::double_mantissa::Infinity<double>(), 8,
              DecimalScientificNotation{
                  true, 0, std::vector<unsigned char>{'i', 'n', 'f'}}),
          std::make_tuple(
              -mantis::double_mantissa::Infinity<double>(), 8,
              DecimalScientificNotation{
                  false, 0, std::vector<unsigned char>{'i', 'n', 'f'}}),
      };

  // The maximum number of digits required for an exact decimal round-trip.
  const int max_digits10 =
      std::numeric_limits<DoubleMantissa<double>>::max_digits10;

  // The double-mantissa epsilon.
  const double epsilon = std::numeric_limits<DoubleMantissa<double>>::epsilon();

  for (const auto& test : tests) {
    const DoubleMantissa<double> value = std::get<0>(test);
    const int num_digits = std::get<1>(test);
    const DecimalScientificNotation expected_rep = std::get<2>(test);

    const DecimalScientificNotation rep =
        value.DecimalScientificNotation(num_digits);
    REQUIRE(rep.positive == expected_rep.positive);
    REQUIRE(rep.exponent == expected_rep.exponent);
    REQUIRE(rep.digits.size() == expected_rep.digits.size());
    for (unsigned digit = 0; digit < rep.digits.size(); ++digit) {
      REQUIRE(rep.digits[digit] == expected_rep.digits[digit]);
    }

    if (std::isfinite(value)) {
      const DecimalScientificNotation full_rep =
          value.DecimalScientificNotation(max_digits10);
      const DoubleMantissa<double> value_reformed(full_rep);

      const DoubleMantissa<double> error = value - value_reformed;
      const double tolerance = epsilon * std::abs(value.Upper());
      REQUIRE(std::abs(error.Upper()) <= tolerance);
    }
  }

  const std::vector<std::pair<std::string, DecimalScientificNotation>>
      string_tests{
          std::make_pair(
              "3.14159",
              DecimalScientificNotation{
                  true, 0,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "+3.14159",
              DecimalScientificNotation{
                  true, 0,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "-3.14159",
              DecimalScientificNotation{
                  false, 0,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "3.14159e2",
              DecimalScientificNotation{
                  true, 2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "3.14159E2",
              DecimalScientificNotation{
                  true, 2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "-3.14159e2",
              DecimalScientificNotation{
                  false, 2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "-3.14159E2",
              DecimalScientificNotation{
                  false, 2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "3.14159e-2",
              DecimalScientificNotation{
                  true, -2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "-3.14159e-2",
              DecimalScientificNotation{
                  false, -2,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "314159",
              DecimalScientificNotation{
                  true, 5,
                  std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                             5_uchar, 9_uchar}}),
          std::make_pair(
              "-314159",
              DecimalScientificNotation{
                  false,
                  5, std::vector<unsigned char>{3_uchar, 1_uchar, 4_uchar,
                                                1_uchar, 5_uchar, 9_uchar}}),
          std::make_pair(
              ".314159",
              DecimalScientificNotation{
                  true, 0,
                  std::vector<unsigned char>{0_uchar, 3_uchar, 1_uchar, 4_uchar,
                                             1_uchar, 5_uchar, 9_uchar}}),
          std::make_pair(
              "+.314159",
              DecimalScientificNotation{true, 0,
                                        std::vector<unsigned char>{
                                            0_uchar, 3_uchar, 1_uchar, 4_uchar,
                                            1_uchar, 5_uchar, 9_uchar}}),
          std::make_pair("-.314159",
                         DecimalScientificNotation{
                             false, 0,
                             std::vector<unsigned char>{
                                 0_uchar, 3_uchar, 1_uchar, 4_uchar, 1_uchar,
                                 5_uchar, 9_uchar}}),

      };

  for (const auto& test : string_tests) {
    const std::string& input = test.first;
    const DecimalScientificNotation& expected_rep = test.second;

    DecimalScientificNotation rep;
    rep.FromString(input);
    REQUIRE(rep.positive == expected_rep.positive);
    REQUIRE(rep.exponent == expected_rep.exponent);
    REQUIRE(rep.digits.size() == expected_rep.digits.size());
    for (unsigned digit = 0; digit < rep.digits.size(); ++digit) {
      REQUIRE(rep.digits[digit] == expected_rep.digits[digit]);
    }
  }
}
