/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_DOUBLE_MANTISSA_STD_NORMAL_DISTRIBUTION_IMPL_H_
#define MANTIS_DOUBLE_MANTISSA_STD_NORMAL_DISTRIBUTION_IMPL_H_

#include "mantis/double_mantissa.hpp"

namespace std {

inline normal_distribution<
    mantis::DoubleMantissa<float>>::param_type::param_type(result_type mean,
                                                           result_type stddev)
    : mean_(mean), stddev_(stddev) {}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::param_type::mean() const {
  return mean_;
}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::param_type::stddev() const {
  return stddev_;
}

inline bool normal_distribution<mantis::DoubleMantissa<float>>::param_type::
operator==(const param_type& right) const {
  return mean_ == right.mean() && stddev_ == right.stddev();
}

inline bool normal_distribution<mantis::DoubleMantissa<float>>::param_type::
operator!=(const param_type& right) const {
  return mean_ != right.mean() || stddev_ != right.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<float>>::normal_distribution(
    result_type mean, result_type stddev)
    : param_(mean, stddev) {}

inline normal_distribution<mantis::DoubleMantissa<float>>::normal_distribution(
    const param_type& param)
    : param_(param) {}

inline void normal_distribution<mantis::DoubleMantissa<float>>::reset() {
  have_saved_result_ = false;
}

template <class URNG>
mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::operator()(URNG& gen) {
  return operator()(gen, param_);
}

template <class URNG>
mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::operator()(
    URNG& gen, const param_type& param) {
  // Generate a sample from the standard normal distribution using Marsaglia's
  // polar method.
  result_type polar_result;
  if (have_saved_result_) {
    have_saved_result_ = false;
    polar_result = saved_result_;
  } else {
    // Uniformly sample from the unit square until a sample is drawn from the
    // unit circle: including the boundary but excluding the center. The result
    // will be uniformly distributed over the kept region.
    result_type x, y, squared_norm;
    std::uniform_real_distribution<result_type> uniform_dist(result_type(-1),
                                                             result_type(1));
    do {
      x = uniform_dist(gen);
      y = uniform_dist(gen);
      squared_norm = x * x + y * y;
    } while (squared_norm > 1 || squared_norm == 0);

    const result_type scale =
        std::sqrt(-2 * std::log(squared_norm) / squared_norm);
    polar_result = x * scale;
    saved_result_ = y * scale;
    have_saved_result_ = true;
  }

  return param.mean() + param.stddev() * polar_result;
}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::mean() const {
  return param_.mean();
}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::stddev() const {
  return param_.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<float>>::param_type
normal_distribution<mantis::DoubleMantissa<float>>::param() const {
  return param_;
}

inline void normal_distribution<mantis::DoubleMantissa<float>>::param(
    const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::min() const {
  return numeric_limits<mantis::DoubleMantissa<float>>::min();
}

inline mantis::DoubleMantissa<float>
normal_distribution<mantis::DoubleMantissa<float>>::max() const {
  return numeric_limits<mantis::DoubleMantissa<float>>::max();
}

inline normal_distribution<
    mantis::DoubleMantissa<double>>::param_type::param_type(result_type mean,
                                                            result_type stddev)
    : mean_(mean), stddev_(stddev) {}

inline mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::param_type::mean() const {
  return mean_;
}

inline mantis::DoubleMantissa<double> normal_distribution<
    mantis::DoubleMantissa<double>>::param_type::stddev() const {
  return stddev_;
}

inline bool normal_distribution<mantis::DoubleMantissa<double>>::param_type::
operator==(const param_type& right) const {
  return mean_ == right.mean() && stddev_ == right.stddev();
}

inline bool normal_distribution<mantis::DoubleMantissa<double>>::param_type::
operator!=(const param_type& right) const {
  return mean_ != right.mean() || stddev_ != right.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<double>>::normal_distribution(
    result_type mean, result_type stddev)
    : param_(mean, stddev) {}

inline normal_distribution<mantis::DoubleMantissa<double>>::normal_distribution(
    const param_type& param)
    : param_(param) {}

inline void normal_distribution<mantis::DoubleMantissa<double>>::reset() {
  have_saved_result_ = false;
}

template <class URNG>
mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::operator()(URNG& gen) {
  return operator()(gen, param_);
}

template <class URNG>
mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::operator()(
    URNG& gen, const param_type& param) {
  // Generate a sample from the standard normal distribution using Marsaglia's
  // polar method.
  result_type polar_result;
  if (have_saved_result_) {
    have_saved_result_ = false;
    polar_result = saved_result_;
  } else {
    // Uniformly sample from the unit square until a sample is drawn from the
    // unit circle: including the boundary but excluding the center. The result
    // will be uniformly distributed over the kept region.
    result_type x, y, squared_norm;
    std::uniform_real_distribution<result_type> uniform_dist(result_type(-1),
                                                             result_type(1));
    do {
      x = uniform_dist(gen);
      y = uniform_dist(gen);
      squared_norm = x * x + y * y;
    } while (squared_norm > 1 || squared_norm == 0);

    const result_type scale =
        std::sqrt(-2 * std::log(squared_norm) / squared_norm);
    polar_result = x * scale;
    saved_result_ = y * scale;
    have_saved_result_ = true;
  }

  return param.mean() + param.stddev() * polar_result;
}

inline mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::mean() const {
  return param_.mean();
}

inline mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::stddev() const {
  return param_.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<double>>::param_type
normal_distribution<mantis::DoubleMantissa<double>>::param() const {
  return param_;
}

inline void normal_distribution<mantis::DoubleMantissa<double>>::param(
    const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::min() const {
  return numeric_limits<mantis::DoubleMantissa<double>>::min();
}

inline mantis::DoubleMantissa<double>
normal_distribution<mantis::DoubleMantissa<double>>::max() const {
  return numeric_limits<mantis::DoubleMantissa<double>>::max();
}

inline normal_distribution<mantis::DoubleMantissa<long double>>::param_type::
    param_type(result_type mean, result_type stddev)
    : mean_(mean), stddev_(stddev) {}

inline mantis::DoubleMantissa<long double> normal_distribution<
    mantis::DoubleMantissa<long double>>::param_type::mean() const {
  return mean_;
}

inline mantis::DoubleMantissa<long double> normal_distribution<
    mantis::DoubleMantissa<long double>>::param_type::stddev() const {
  return stddev_;
}

inline bool
normal_distribution<mantis::DoubleMantissa<long double>>::param_type::
operator==(const param_type& right) const {
  return mean_ == right.mean() && stddev_ == right.stddev();
}

inline bool
normal_distribution<mantis::DoubleMantissa<long double>>::param_type::
operator!=(const param_type& right) const {
  return mean_ != right.mean() || stddev_ != right.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<long double>>::
    normal_distribution(result_type mean, result_type stddev)
    : param_(mean, stddev) {}

inline normal_distribution<mantis::DoubleMantissa<long double>>::
    normal_distribution(const param_type& param)
    : param_(param) {}

inline void normal_distribution<mantis::DoubleMantissa<long double>>::reset() {
  have_saved_result_ = false;
}

template <class URNG>
mantis::DoubleMantissa<long double> normal_distribution<
    mantis::DoubleMantissa<long double>>::operator()(URNG& gen) {
  return operator()(gen, param_);
}

template <class URNG>
mantis::DoubleMantissa<long double>
normal_distribution<mantis::DoubleMantissa<long double>>::operator()(
    URNG& gen, const param_type& param) {
  // Generate a sample from the standard normal distribution using Marsaglia's
  // polar method.
  result_type polar_result;
  if (have_saved_result_) {
    have_saved_result_ = false;
    polar_result = saved_result_;
  } else {
    // Uniformly sample from the unit square until a sample is drawn from the
    // unit circle: including the boundary but excluding the center. The result
    // will be uniformly distributed over the kept region.
    result_type x, y, squared_norm;
    std::uniform_real_distribution<result_type> uniform_dist(result_type(-1),
                                                             result_type(1));
    do {
      x = uniform_dist(gen);
      y = uniform_dist(gen);
      squared_norm = x * x + y * y;
    } while (squared_norm > 1 || squared_norm == 0);

    const result_type scale =
        std::sqrt(-2 * std::log(squared_norm) / squared_norm);
    polar_result = x * scale;
    saved_result_ = y * scale;
    have_saved_result_ = true;
  }

  return param.mean() + param.stddev() * polar_result;
}

inline mantis::DoubleMantissa<long double>
normal_distribution<mantis::DoubleMantissa<long double>>::mean() const {
  return param_.mean();
}

inline mantis::DoubleMantissa<long double>
normal_distribution<mantis::DoubleMantissa<long double>>::stddev() const {
  return param_.stddev();
}

inline normal_distribution<mantis::DoubleMantissa<long double>>::param_type
normal_distribution<mantis::DoubleMantissa<long double>>::param() const {
  return param_;
}

inline void normal_distribution<mantis::DoubleMantissa<long double>>::param(
    const param_type& param) {
  param_ = param;
}

inline mantis::DoubleMantissa<long double>
normal_distribution<mantis::DoubleMantissa<long double>>::min() const {
  return numeric_limits<mantis::DoubleMantissa<long double>>::min();
}

inline mantis::DoubleMantissa<long double>
normal_distribution<mantis::DoubleMantissa<long double>>::max() const {
  return numeric_limits<mantis::DoubleMantissa<long double>>::max();
}

}  // namespace std

#endif  // ifndef MANTIS_DOUBLE_MANTISSA_STD_NORMAL_DISTRIBUTION_IMPL_H_
