/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_BINARY_NOTATION_IMPL_H_
#define MANTIS_BINARY_NOTATION_IMPL_H_

#include "mantis/binary_notation.hpp"

namespace mantis {

inline std::string BinaryNotation::ToString() const {
  std::string s;
  if (!positive) {
    s += '-';
  }
  if (digits.size() == 3 && digits[0] == 'i') {
    s += "inf";
  } else if (digits.size() == 3 && digits[0] == 'n') {
    s += "nan";
  } else {
    s += "0.";
    for (unsigned digit = 0; digit < digits.size(); ++digit) {
      s += std::to_string(unsigned(digits[digit]));
    }
    if (exponent != 0) {
      s += 'e';
      s += std::to_string(exponent);
    }
  }
  return s;
}

inline BinaryNotation& BinaryNotation::FromString(const std::string& rep) {
  positive = true;
  exponent = 0;
  digits.clear();
  if (rep.empty()) {
    return *this;
  }

  std::string::size_type value_pos = 0;
  if (rep[0] == '-') {
    positive = false;
    value_pos = 1;
  } else if (rep[0] == '+') {
    value_pos = 1;
  }

  const std::string::size_type lower_exp_pos = rep.find('e', value_pos);
  const std::string::size_type upper_exp_pos = rep.find('E', value_pos);
  const std::string::size_type exp_pos =
      lower_exp_pos != std::string::npos ? lower_exp_pos : upper_exp_pos;

  std::string value_string;
  if (exp_pos != std::string::npos) {
    value_string = rep.substr(value_pos, exp_pos - value_pos);
  } else {
    value_string = rep.substr(value_pos);
  }

  unsigned num_digits;
  const std::string::size_type period_pos = value_string.find('.');
  const bool have_period = period_pos != std::string::npos;
  if (have_period) {
    num_digits = value_string.size() - 1;
    digits.resize(num_digits);
    if (period_pos == 0) {
      // The number is of the form .0101, so there is an implicit zero.
      exponent = 0;
      for (unsigned digit = 0; digit < num_digits; ++digit) {
        digits[digit] = value_string[digit + 1] - '0';
      }
    } else {
      exponent = period_pos;
      for (unsigned digit = 0; digit < num_digits; ++digit) {
        if (digit < period_pos) {
          digits[digit] = value_string[digit] - '0';
        } else {
          digits[digit] = value_string[digit + 1] - '0';
        }
      }
    }
  } else {
    num_digits = value_string.size();
    exponent = num_digits;
    digits.resize(num_digits);
    for (unsigned digit = 0; digit < num_digits; ++digit) {
      digits[digit] = value_string[digit] - '0';
    }
  }

  // TODO(Jack Poulson): Ensure each digit is in [0, 1].

  if (exp_pos != std::string::npos) {
    const std::string exp_string = rep.substr(exp_pos + 1);
    exponent += std::stoi(exp_string);
  }

  return *this;
}

}  // namespace mantis

#endif  // ifndef MANTIS_BINARY_NOTATION_IMPL_H_
