AC_INIT([sketchrec], 0.1)
AC_PREREQ(2.69)

AC_LANG([C++])

AM_INIT_AUTOMAKE([1.12 foreign no-define nostdinc])

: ${CXXFLAGS="-pedantic -Wall -g -O3 -funroll-loops"}

# Check for programs.
AC_PROG_CXX
AC_PROG_CXXCPP
PKG_PROG_PKG_CONFIG

# Check for libraries.
PKG_CHECK_MODULES(CAIRO, [cairo >= 1.10])
PKG_CHECK_MODULES(FFTW, [fftw3f >= 3.0])
PKG_CHECK_MODULES(GLIB, [glib-2.0 >= 2.0])
PKG_CHECK_MODULES(GTKMM, [gtkmm-2.4 >= 2.24])
PKG_CHECK_MODULES(LIBRSVG, [librsvg-2.0 >= 2.0])

# AX_LIB_DLIB([MIN-VERSION],[ACTION-IF-SUCCESS],[ACTION-IF-FAILURE])
# ------------------------------------------------------------------
# Check for dlib with at least major version MIN-VERSION.  On success, set
# HAVE_DLIB and execute ACTION-IF-SUCCESS, otherwise execute
# ACTION-IF-FAILURE.
AC_DEFUN([AX_LIB_DLIB], [
  AC_CACHE_CHECK([for dlib], [ax_cv_have_dlib], [
    ax_cv_have_dlib=no
    AC_LANG_PUSH([C++])
    AC_PREPROC_IFELSE([
      AC_LANG_PROGRAM([
        #include <dlib/matrix.h>
        #include <dlib/revision.h>
        #include <dlib/svm.h>
        #include <dlib/type_safe_union.h>
        #include <dlib/unordered_pair.h>
        #if !defined(DLIB_MAJOR_VERSION) || DLIB_MAJOR_VERSION < $1
          #error
        #endif
      ])
    ], [ax_cv_have_dlib=yes], [])
    AC_LANG_POP
  ])
  if test "$ax_cv_have_dlib" = yes; then
    AC_DEFINE([HAVE_DLIB], [1], [Define if dlib >= $1 is present.])
    $2
  else
    $3
  fi
])

AX_LIB_DLIB([18], , [AC_MSG_FAILURE([dlib 18 or newer is required])])

# Check for compiler characteristics.
AC_HEADER_ASSERT
AC_OPENMP
AX_CXX_COMPILE_STDCXX_11([noext], [mandatory])

AC_CONFIG_FILES([Makefile src/Makefile])

AC_OUTPUT
