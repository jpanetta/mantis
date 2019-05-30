/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_ENABLE_IF_H_
#define MANTIS_ENABLE_IF_H_

#include <type_traits>

namespace mantis {

// For overloading function definitions using type traits. For example:
//
//   template<typename T, typename=EnableIf<std::is_integral<T>>>
//   int Foo(T value);
//
//   template<typename T, typename=DisableIf<std::is_integral<T>>>
//   double Foo(T value);
//
// would lead to the 'Foo' function returning an 'int' for any integral type
// and a 'double' for any non-integral type.
template <typename Condition, class T = void>
using EnableIf = typename std::enable_if<Condition::value, T>::type;

template <typename Condition, class T = void>
using DisableIf = typename std::enable_if<!Condition::value, T>::type;

}  // namespace mantis

#endif  // ifndef MANTIS_ENABLE_IF_H_
