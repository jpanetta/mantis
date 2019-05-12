[**mantis**](https://gitlab.com/hodge_star/mantis) is a
[C++14](https://en.wikipedia.org/wiki/C%2B%2B14), header-only implementation of
double-mantissa arithmetic.

[![Join the chat at https://gitter.im/hodge_star/community](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/hodge_star/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

### Dependencies
There are no dependencies beyond C++14 for manually including mantis's headers
into your project.

But, if you would like to make use of the project's build system, the only
strict dependency is:

* [meson](http://mesonbuild.com): "Meson is an open source build system meant
to be both extremely fast, and, even more importantly, as user friendly as
possible."

Meson will automatically install
[Catch2](https://github.com/catchorg/Catch2) (a header-only C++
unit-testing library) and [specify](https://gitlab.com/hodge_star/specify)
(a C++14 header-only, command-line argument processor).
