/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_STD_RANDOM_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_STD_RANDOM_IMPL_H_

#include "mantis/double_mantissa.hpp"

namespace std {

inline uniform_real_distribution<
    mantis::DoubleMantissa<float>>::param_type::param_type(result_type a_val,
                                                           result_type b_val)
    : a_(a_val), b_(b_val) {}

inline mantis::DoubleMantissa<float> uniform_real_distribution<
    mantis::DoubleMantissa<float>>::param_type::a() const {
  return a_;
}

inline mantis::DoubleMantissa<float> uniform_real_distribution<
    mantis::DoubleMantissa<float>>::param_type::b() const {
  return b_;
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<float>>::param_type::
operator==(const param_type& right) const {
  return a_ == right.a() && b_ == right.b();
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<float>>::param_type::
operator!=(const param_type& right) const {
  return a_ != right.a() || b_ != right.b();
}

inline uniform_real_distribution<
    mantis::DoubleMantissa<float>>::uniform_real_distribution(result_type a,
                                                              result_type b)
    : param_(a, b) {}

inline uniform_real_distribution<mantis::DoubleMantissa<float>>::
    uniform_real_distribution(const param_type& param)
    : param_(param) {}

inline void uniform_real_distribution<mantis::DoubleMantissa<float>>::reset() {}

template <class URNG>
mantis::DoubleMantissa<float> uniform_real_distribution<
    mantis::DoubleMantissa<float>>::operator()(URNG& gen) {
  return a() + result_type::UniformRandom(gen) * (b() - a());
}

template <class URNG>
mantis::DoubleMantissa<float>
uniform_real_distribution<mantis::DoubleMantissa<float>>::operator()(
    URNG& gen, const param_type& param) {
  return param.a() + result_type::UniformRandom(gen) * (param.b() - param.a());
}

inline mantis::DoubleMantissa<float>
uniform_real_distribution<mantis::DoubleMantissa<float>>::a() const {
  return param_.a();
}

inline mantis::DoubleMantissa<float>
uniform_real_distribution<mantis::DoubleMantissa<float>>::b() const {
  return param_.b();
}

inline uniform_real_distribution<mantis::DoubleMantissa<float>>::param_type
uniform_real_distribution<mantis::DoubleMantissa<float>>::param() const {
  return param_;
}

inline void uniform_real_distribution<mantis::DoubleMantissa<float>>::param(
    const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<float>
uniform_real_distribution<mantis::DoubleMantissa<float>>::min() const {
  return param_.a();
}

inline mantis::DoubleMantissa<float>
uniform_real_distribution<mantis::DoubleMantissa<float>>::max() const {
  return param_.b();
}

inline uniform_real_distribution<
    mantis::DoubleMantissa<double>>::param_type::param_type(result_type a_val,
                                                            result_type b_val)
    : a_(a_val), b_(b_val) {}

inline mantis::DoubleMantissa<double> uniform_real_distribution<
    mantis::DoubleMantissa<double>>::param_type::a() const {
  return a_;
}

inline mantis::DoubleMantissa<double> uniform_real_distribution<
    mantis::DoubleMantissa<double>>::param_type::b() const {
  return b_;
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<double>>::param_type::
operator==(const param_type& right) const {
  return a_ == right.a() && b_ == right.b();
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<double>>::param_type::
operator!=(const param_type& right) const {
  return a_ != right.a() || b_ != right.b();
}

inline uniform_real_distribution<
    mantis::DoubleMantissa<double>>::uniform_real_distribution(result_type a,
                                                               result_type b)
    : param_(a, b) {}

inline uniform_real_distribution<mantis::DoubleMantissa<double>>::
    uniform_real_distribution(const param_type& param)
    : param_(param) {}

inline void uniform_real_distribution<mantis::DoubleMantissa<double>>::reset() {
}

template <class URNG>
mantis::DoubleMantissa<double> uniform_real_distribution<
    mantis::DoubleMantissa<double>>::operator()(URNG& gen) {
  return a() + result_type::UniformRandom(gen) * (b() - a());
}

template <class URNG>
mantis::DoubleMantissa<double>
uniform_real_distribution<mantis::DoubleMantissa<double>>::operator()(
    URNG& gen, const param_type& param) {
  return param.a() + result_type::UniformRandom(gen) * (param.b() - param.a());
}

inline mantis::DoubleMantissa<double>
uniform_real_distribution<mantis::DoubleMantissa<double>>::a() const {
  return param_.a();
}

inline mantis::DoubleMantissa<double>
uniform_real_distribution<mantis::DoubleMantissa<double>>::b() const {
  return param_.b();
}

inline uniform_real_distribution<mantis::DoubleMantissa<double>>::param_type
uniform_real_distribution<mantis::DoubleMantissa<double>>::param() const {
  return param_;
}

inline void uniform_real_distribution<mantis::DoubleMantissa<double>>::param(
    const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<double>
uniform_real_distribution<mantis::DoubleMantissa<double>>::min() const {
  return param_.a();
}

inline mantis::DoubleMantissa<double>
uniform_real_distribution<mantis::DoubleMantissa<double>>::max() const {
  return param_.b();
}

inline uniform_real_distribution<mantis::DoubleMantissa<long double>>::
    param_type::param_type(result_type a_val, result_type b_val)
    : a_(a_val), b_(b_val) {}

inline mantis::DoubleMantissa<long double> uniform_real_distribution<
    mantis::DoubleMantissa<long double>>::param_type::a() const {
  return a_;
}

inline mantis::DoubleMantissa<long double> uniform_real_distribution<
    mantis::DoubleMantissa<long double>>::param_type::b() const {
  return b_;
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<long double>>::param_type::
operator==(const param_type& right) const {
  return a_ == right.a() && b_ == right.b();
}

inline bool
uniform_real_distribution<mantis::DoubleMantissa<long double>>::param_type::
operator!=(const param_type& right) const {
  return a_ != right.a() || b_ != right.b();
}

inline uniform_real_distribution<mantis::DoubleMantissa<long double>>::
    uniform_real_distribution(result_type a, result_type b)
    : param_(a, b) {}

inline uniform_real_distribution<mantis::DoubleMantissa<long double>>::
    uniform_real_distribution(const param_type& param)
    : param_(param) {}

inline void
uniform_real_distribution<mantis::DoubleMantissa<long double>>::reset() {}

template <class URNG>
mantis::DoubleMantissa<long double> uniform_real_distribution<
    mantis::DoubleMantissa<long double>>::operator()(URNG& gen) {
  return a() + result_type::UniformRandom(gen) * (b() - a());
}

template <class URNG>
mantis::DoubleMantissa<long double>
uniform_real_distribution<mantis::DoubleMantissa<long double>>::operator()(
    URNG& gen, const param_type& param) {
  return param.a() + result_type::UniformRandom(gen) * (param.b() - param.a());
}

inline mantis::DoubleMantissa<long double>
uniform_real_distribution<mantis::DoubleMantissa<long double>>::a() const {
  return param_.a();
}

inline mantis::DoubleMantissa<long double>
uniform_real_distribution<mantis::DoubleMantissa<long double>>::b() const {
  return param_.b();
}

inline uniform_real_distribution<
    mantis::DoubleMantissa<long double>>::param_type
uniform_real_distribution<mantis::DoubleMantissa<long double>>::param() const {
  return param_;
}

inline void uniform_real_distribution<
    mantis::DoubleMantissa<long double>>::param(const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<long double>
uniform_real_distribution<mantis::DoubleMantissa<long double>>::min() const {
  return param_.a();
}

inline mantis::DoubleMantissa<long double>
uniform_real_distribution<mantis::DoubleMantissa<long double>>::max() const {
  return param_.b();
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_STD_RANDOM_IMPL_H_
