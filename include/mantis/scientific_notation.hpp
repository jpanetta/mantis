/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_SCIENTIFIC_NOTATION_H_
#define MANTIS_SCIENTIFIC_NOTATION_H_

#include <string>
#include <vector>

namespace mantis {

// A structure for the decimal scientific notation representation of a
// floating-point value. It makes use of the standard scientific notation:
//
//   +- d_0 . d_1 d_2 ... d_{n - 1} x 10^{exponent},
//
// where each d_j lives in [0, 9] and 'exponent' is an integer.
// We contiguously store the decimal digits with this implied format. For
// example, to represent the first several digits of pi, we would have:
//
//   positive: true,
//   exponent: 0,
//   d: [3, 1, 4, 1, 5, 9, ...]
//
// To represent -152.4, we would have
//
//   positive: false,
//   exponent: 2,
//   d: [1, 5, 2, 4]
//
// The special value of NaN is handled via:
//
//   positive: true,
//   exponent: 0,
//   d: ['n', 'a', 'n'].
//
// Similarly, infinity is handled via:
//
//   positive: true,
//   exponent: 0,
//   d: ['i', 'n', 'f'],
//
// and negative infinity has 'positive' equal to false.
//
struct ScientificNotation {
  // The sign of the value.
  bool positive = true;

  // The exponent of the scientific notation of the value.
  int exponent = 0;

  // Each entry contains a value in the range 0 to 9, except in the cases of
  // NaN and +-infinity.
  std::vector<unsigned char> digits;

  // Returns a string for the decimal scientific notation.
  std::string ToString() const;

  // Fills this class by converting a string in decimal scientific notation.
  ScientificNotation& FromString(const std::string& rep);
};

}  // namespace std

#include "mantis/scientific_notation-impl.hpp"

#endif  // ifndef MANTIS_SCIENTIFIC_NOTATION_H_
