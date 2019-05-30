/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_MACROS_H_
#define MANTIS_MACROS_H_

#include <iostream>

// An attribute for routines which are known to not throw exceptions in
// release mode.
#ifdef MANTIS_DEBUG
#define MANTIS_NOEXCEPT
#else
#define MANTIS_NOEXCEPT noexcept
#endif

#ifdef MANTIS_DEBUG
#define MANTIS_ASSERT(assertion, message)  \
  {                                        \
    if (!(assertion)) {                    \
      std::cerr << (message) << std::endl; \
    }                                      \
  }
#else
#define MANTIS_ASSERT(condition, message)
#endif

#ifdef __GNUG__
#define MANTIS_UNUSED __attribute__((unused))
#elif defined(__clang__)
#define MANTIS_UNUSED __attribute__((unused))
#else
#define MANTIS_UNUSED
#endif  // ifdef __GNUG__

// We have not yet enabled C++20.
#define CXX20_CONSTEXPR

#endif  // ifndef MANTIS_MACROS_H_
