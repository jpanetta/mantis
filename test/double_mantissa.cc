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
  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(-3., 4e-20);
  const DoubleMantissa<double> z = mantis::Square(x);

  const DoubleMantissa<double> z_expected(9., -2.4e-19);
  const DoubleMantissa<double> z_error = z - z_expected;
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z_expected.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z_expected.Lower()));
}

TEST_CASE("Divide", "[Divide]") {
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
  const double eps = std::numeric_limits<double>::epsilon();
  const DoubleMantissa<double> x(3., 4e-20), y(4., 5e-20), z(0., 0.);
  const DoubleMantissa<double> x_sqrt = std::sqrt(x);
  const DoubleMantissa<double> y_sqrt = std::sqrt(y);
  const DoubleMantissa<double> z_sqrt = std::sqrt(z);

  const DoubleMantissa<double> x_error = x - mantis::Square(x_sqrt);
  const DoubleMantissa<double> y_error = y - mantis::Square(y_sqrt);
  const DoubleMantissa<double> z_error = z - mantis::Square(z_sqrt);
  REQUIRE(std::abs(x_error.Upper()) <= 2 * eps * std::abs(x.Upper()));
  REQUIRE(std::abs(x_error.Lower()) <= 2 * eps * std::abs(x.Lower()));
  REQUIRE(std::abs(y_error.Upper()) <= 2 * eps * std::abs(y.Upper()));
  REQUIRE(std::abs(y_error.Lower()) <= 2 * eps * std::abs(y.Lower()));
  REQUIRE(std::abs(z_error.Upper()) <= 2 * eps * std::abs(z.Upper()));
  REQUIRE(std::abs(z_error.Lower()) <= 2 * eps * std::abs(z.Lower()));

  const DoubleMantissa<double> neg_sqrt =
      std::sqrt(DoubleMantissa<double>(-1., 0.));
  REQUIRE(neg_sqrt.Upper() != neg_sqrt.Upper());
}
