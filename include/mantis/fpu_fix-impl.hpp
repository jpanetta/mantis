/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_FPU_FIX_IMPL_H_
#define MANTIS_FPU_FIX_IMPL_H_

#include "mantis/fpu_fix.hpp"

namespace mantis {

// TODO(Jack Poulson): Handle other architectural/system combinations.
// We follow the basic approach of the QD library of Hida et al.
#if defined(X86) && defined(LINUX)

#define FPU_DOUBLE 0x0200
#define FPU_EXTENDED 0x0300

#define FPU_GETCW(x) asm volatile("fnstcw %0" : "=m"(x));
#define FPU_SETCW(x) asm volatile("fldcw %0" : : "m"(x));

#endif  // if defined(X86) && defined(LINUX)

inline FPUFix::FPUFix() {
#if defined(X86) && defined(LINUX)
  volatile unsigned short control_word, new_control_word;
  FPU_GETCW(control_word);
  new_control_word = (control_word & ~FPU_EXTENDED) | FPU_DOUBLE;
  FPU_SETCW(new_control_word);
  control_word_ = control_word;
#endif  // if defined(X86) && defined(LINUX)
}

inline FPUFix::~FPUFix() {
#if defined(X86) && defined(LINUX)
  FPU_SETCW(control_word_);
#endif  // if defined(X86) && defined(LINUX)
}

}  // namespace mantis

#endif  // ifndef MANTIS_FPU_FIX_IMPL_H_
