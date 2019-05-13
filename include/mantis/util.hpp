/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef MANTIS_UTIL_H_
#define MANTIS_UTIL_H_

namespace mantis {

// Performs a -- preferably fused -- multiply and add of the form: x * y + z.
template <typename Real>
Real MultiplyAdd(const Real& x, const Real& y, const Real& z);

// A specialization of the Multiply Add to fp32.
float MultiplyAdd(const float& x, const float& y, const float& z);

// A specialization of the Multiply Add to fp64.
double MultiplyAdd(const double& x, const double& y, const double& z);

// A specialization of the Multiply Add to long double.
long double MultiplyAdd(const long double& x, const long double& y,
                        const long double& z);

// Performs a -- preferably fused -- multiply subtract: x * y - z.
template <typename Real>
Real MultiplySubtract(const Real& x, const Real& y, const Real& z);

// A specialization of the Multiply Subtract to fp32.
float MultiplySubtract(const float& x, const float& y, const float& z);

// A specialization of the Multiply Subtract to fp64.
double MultiplySubtract(const double& x, const double& y, const double& z);

// A specialization of the Multiply Subtract to long double.
long double MultiplySubtract(const long double& x, const long double& y,
                             const long double& z);

// Returns fl(larger + smaller) and stores error := err(larger + smaller),
// assuming |larger| >= |smaller|. The result and the error are computed with
// a total of three floating-point operations; this approach is faster, but
// the error approximation is less accurate, than that of TwoSum.
//
// Please see [1, Alg. 3] and [2, pg. 312] for details.
template <typename Real>
Real QuickTwoSum(const Real& larger, const Real& smaller, Real* error);

// Returns fl(larger + smaller) and stores error := err(larger + smaller)
// using six floating-point operations. The result and the error are computed
// with a total of six floating-point operations; this approach is slower, but
// the error approximation is more accurate, than that of QuickTwoSum.
//
// Please see [1, Alg. 4] and [2, pg. 314] for details.
template <typename Real>
Real TwoSum(const Real& larger, const Real& smaller, Real* error);

// Returns fl(larger - smaller) and stores error := err(larger - smaller),
// assuming |larger| >= |smaller|. The result and the error are computed with
// a total of three floating-point operations; this approach is faster, but
// the error approximation is less accurate, than that of TwoDiff.
//
// Please see [1, Alg. 3] and [2, pg. 312] for details on the additive
// equivalent.
template <typename Real>
Real QuickTwoDiff(const Real& larger, const Real& smaller, Real* error);

// Returns fl(larger - smaller) and stores error := err(larger - smaller)
// using six floating-point operations. The result and the error are computed
// with a total of six floating-point operations; this approach is slower, but
// the error approximation is more accurate, than that of QuickTwoDiff.
//
// Please see [1, Alg. 4] and [2, pg. 314] for details on the additive
// equivalent.
template <typename Real>
Real TwoDiff(const Real& larger, const Real& smaller, Real* error);

// Generalize the 'SPLIT' algorithm from [1, Alg. 5] and [2, pg. 325], as
// implemented in the QD library of Hida et al.
template <typename Real>
void Split(const Real& value, Real* high, Real* low);

// Returns fl(x * y) and stores error := err(x * y) using an FMA. For some
// datatypes, such as float, double, and long double, this FMA is guaranteed
// to be fl(x * y + z) rather than fl(fl(x + y) + z).
//
// Please see [1, Alg. 7] for details.
template <typename Real>
Real TwoProdFMA(const Real& x, const Real& y, Real* error);

// Returns fl(x * y) and stores error := err(x * y). For datatypes where an
// accurate FMA can be computed, such as float, double, and long double, the
// FMA approach is used. Otherwise, the 'Split' approach is used.
//
// Please see [1, Alg. 6] and [2, pg. 326] for details on the 'Split' approach.
template <typename Real>
Real TwoProd(const Real& x, const Real& y, Real* error);
float TwoProd(const float& x, const float& y, float* error);
double TwoProd(const double& x, const double& y, double* error);
long double TwoProd(const long double& x, const long double& y,
                    long double* error);

// A specialization of TwoProd to the input arguments being equal.
template <typename Real>
Real TwoSquare(const Real& x, Real* error);
float TwoSquare(const float& x, float* error);
double TwoSquare(const double& x, double* error);
long double TwoSquare(const long double& x, long double* error);

class FPUFix {
 public:
  FPUFix();

  ~FPUFix();

 private:
  unsigned int control_word_ = 0;
};

// References:
//
// [1] Yozo Hida, Xiaoye S. Li, and David H. Bailey,
//     "Library for Double-Double and Quad-Double Arithmetic",
//     Technical Report, NERSC Division, Lawrence Berkeley National Laboratory,
//     USA, 2007.
//
// [2] Jonathan R. Shewchuk,
//     "Adaptive precision floating-point arithmetic and fast robust geometric
//     predicates.", Discrete & Computational Geometry, 18(3):305--363, 1997.
//

}  // namespace mantis

#include "mantis/util-impl.hpp"

#endif  // ifndef MANTIS_UTIL_H_
