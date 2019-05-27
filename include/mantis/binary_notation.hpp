/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_BINARY_NOTATION_H_
#define MANTIS_BINARY_NOTATION_H_

#include <string>
#include <vector>

#include "mantis/decimal_notation.hpp"

namespace mantis {

// A structure for the binary notation representation of a floating-point value.
// It makes use of the standard binary notation:
//
//   +- 0 . d_0 d_1 ... d_{n - 1} x 2^{exponent},
//
// where each d_j lives in [0, 1] and 'exponent' is an integer.
// We contiguously store the binary digits with this implied format. For
// example, to represent the first several digits of 256, we might have:
//
//   positive: true,
//   exponent: 9
//   d: [1, 0, 0, 0, 0, ...]
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
struct BinaryNotation {
  // The sign of the value.
  bool positive = true;

  // The exponent of the binary notation of the value.
  int exponent = 0;

  // Each entry contains a value in the range 0 to 1, except in the cases of
  // NaN and +-infinity.
  std::vector<unsigned char> digits;

  // Returns a string for the binary notation.
  std::string ToString() const;

  // Fills this class by converting a string in binary notation.
  BinaryNotation& FromString(const std::string& rep);
};

}  // namespace std

#include "mantis/binary_notation-impl.hpp"

#endif  // ifndef MANTIS_BINARY_NOTATION_H_
