/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_FPU_FIX_H_
#define MANTIS_FPU_FIX_H_

namespace mantis {

// A class which, when instantiated, is meant to avoid the usage of extra-
// precision registers for fp64 arithmetic, which could interfere with the
// correctness of double-fp64 arithmetic.
class FPUFix {
 public:
  // Initializes the FPU fix.
  FPUFix();

  // Destructs the FPU fix.
  ~FPUFix();

 private:
  // The control word to be restored upon destruction.
  unsigned short control_word_ = 0;
};

}  // namespace mantis

#include "mantis/fpu_fix-impl.hpp"

#endif  // ifndef MANTIS_FPU_FIX_H_
