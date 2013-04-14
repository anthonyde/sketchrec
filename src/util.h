#ifndef UTIL_H
#define UTIL_H

#include <cassert>
#include <random>
#include <vector>

#include <dlib/matrix.h>

// 3x3 Sobel filter kernels
extern const dlib::matrix< float, 3, 3 > sobel_x, sobel_y;

// Normalize a vector using the L1-norm.
template< class Exp >
const dlib::matrix_op< dlib::op_normalize< Exp > > l1_normalize(
  const dlib::matrix_exp< Exp > &m) {
  typedef dlib::op_normalize< Exp > op;

  typename Exp::type s = sum(m);
  if (s != 0.)
    s = 1. / s;

  return dlib::matrix_op< op >(op(m.ref(), s));
}

// Convert cartesian x- and y-magnitude images to radial magnitude and
// orientation images.
template< class T, long NR1, long NC1, long NR2, long NC2, long NR3, long NC3,
  long NR4, long NC4>
void cart2polar(const dlib::matrix< T, NR1, NC1 > &x,
  const dlib::matrix< T, NR2, NC2 > &y, dlib::matrix< T, NR3, NC3 > &r,
  dlib::matrix< T, NR4, NC4 > &theta) {
  const long rows = x.nr();
  const long cols = x.nc();
  assert(y.nr() == rows);
  assert(y.nc() == cols);
  r.set_size(rows, cols);
  theta.set_size(rows, cols);

  for (long j = 0; j < rows; ++j) {
    for (long i = 0; i < cols; ++i) {
      r(j, i) = std::hypot(x(j, i), y(j, i));
      theta(j, i) = std::atan2(y(j, i), x(j, i));
    }
  }
}

// A stream sampling algorithm for choosing n elements from a stream uniformly
// at random
template< class T >
struct stream_sample {
  typedef typename std::vector< T >::size_type size_type;

  stream_sample(size_type n_) : n(n_), i(0) {
    samples.reserve(n);
  }

  template< class Generator >
  void push_back(Generator &g, const T& x) {
    if (i < n) {
      // Select the first n elements.
      samples.push_back(x);
    }
    else {
      // Select the remaining elements with probability n / (i + 1).
      std::uniform_int_distribution< typename std::vector< T >::size_type >
        uniform_i(0, i);
      if (uniform_i(g) < n) {
        std::uniform_int_distribution< typename std::vector< T >::size_type >
          uniform_n(0, n - 1);
        samples[uniform_n(g)] = x;
      }
    }
    ++i;
  }

  const std::vector< T > &get() const {
    return samples;
  }

private:
  const size_type n;
  size_type i;
  std::vector< T > samples;
};

#endif
