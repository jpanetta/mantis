#
# Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
project(
  'mantis',
  'cpp',
  version : '0.1',
  license : 'MPLv2.0',
  meson_version : '>=0.46.0',
  default_options : ['cpp_std=c++14'])
include_dir = include_directories('include')
cxx = meson.get_compiler('cpp')

# A list of dependencies for building the library.
deps = []

# A list of dependencies for building the unit tests.
test_deps = []

# A list of dependencies for building the example drivers.
example_deps = []

# A list of C++ command-line arguments.
cxx_args = []

cpu_family = build_machine.cpu_family()
if cpu_family == 'x86' or cpu_family == 'x86_64'
  cxx_args += '-DX86'
endif

system = build_machine.system()
if system == 'linux'
  cxx_args += '-DLINUX'
endif

# Handle fast (Fortran-like) std::complex arithmetic.
if cxx.get_id() == 'gcc'
  message('Using gcc')
  # Complex arithmetic is greatly accelerated by adopting the Fortran
  # division and NaN-checking conventions.
  cxx_args += '-fcx-fortran-rules'
endif

# Handle the Fused Multiply Add/Subtraction approaches.
have_fma_intrinsic = false
if not have_fma_intrinsic and cxx.get_id() == 'gcc' and cxx.version() >= '4.3'
  if not get_option('disable_gcc_fma')
    have_fma_intrinsic = true
    cxx_args += '-DMANTIS_GCC_FUSED_MULTIPLY_ADD'
  else
    message('Avoiding GCC __builtin_fma due to disable_gcc_fma option.')
  endif
endif
if not have_fma_intrinsic and cxx.get_id() == 'xlc'
  if not get_option('disable_xlc_fma')
    have_fma_intrinsic = true
    cxx_args += '-DMANTIS_XLC_FUSED_MULTIPLY_ADD'
  else
    message('Avoiding XLC __fmadd due to disable_xlc_fma option.')
  endif
endif

if get_option('ieee_sum')
  # Use the more accurate, but slower, summation mechanism.
  cxx_args += '-DMANTIS_IEEE_SUM'
endif

# Test if specify is installed.
found_specify = cxx.has_header('specify.hpp')
if not found_specify
  specify_proj = subproject('specify')
  specify_dep = specify_proj.get_variable('specify_dep')
  example_deps += specify_dep
endif

# Add Catch for C++ unit tests.
test_deps += subproject('catch2').get_variable('catch2_dep')

double_mantissa_test_exe = executable(
    'double_mantissa_test',
    ['test/double_mantissa.cc', 'include/mantis.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)
test('Double mantissa tests', double_mantissa_test_exe)

# A basic usage of Mantis.
basic_exe = executable(
    'basic',
    ['example/basic.cc', 'include/mantis.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)
