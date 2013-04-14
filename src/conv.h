#ifndef CONV_H
#define CONV_H

#include <complex>
#include <iostream>

#include <dlib/matrix.h>
#include <fftw3.h>

template< class T >
struct fftw_helper;

template<>
struct fftw_helper< float > {
  typedef fftwf_plan plan_type;

  static inline plan_type plan_dft_r2c_2d(int n0, int n1, float *in,
    fftwf_complex *out, unsigned flags) {
    return fftwf_plan_dft_r2c_2d(n0, n1, in, out, flags);
  }

  static inline plan_type plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in,
    float *out, unsigned flags) {
    return fftwf_plan_dft_c2r_2d(n0, n1, in, out, flags);
  }

  static inline void execute_dft_r2c(const plan_type p, float *in,
    fftwf_complex *out) {
    fftwf_execute_dft_r2c(p, in, out);
  }

  static inline void execute_dft_c2r(const plan_type p, fftwf_complex *in,
    float *out) {
    fftwf_execute_dft_c2r(p, in, out);
  }
};

// A class for FFT-based convolution with a fixed kernel
template< class T, long NR, long NC, bool Verbose = false >
struct conv_fft {
  conv_fft(const dlib::matrix< T, NR, NC > &h) {
    dlib::matrix< T, NR, NC > x;
    dlib::matrix< std::complex< T >, NR, NC_Comp > xf;

    plan = fftw_helper< T >::plan_dft_r2c_2d(NR, NC, &x(0, 0),
      reinterpret_cast< T (*)[2] >(&xf(0, 0)), FFTW_PATIENT);
    inv_plan = fftw_helper< T >::plan_dft_c2r_2d(NR, NC,
      reinterpret_cast< T (*)[2] >(&xf(0, 0)), &x(0, 0), FFTW_PATIENT);

    if (Verbose) {
      std::cout << "FFT plan:\n";
      fftwf_print_plan(plan);
      std::cout << '\n';

      std::cout << "Inverse FFT plan:\n";
      fftwf_print_plan(inv_plan);
      std::cout << '\n';
    }

    // Store the transformed kernel.
    fftw_helper< T >::execute_dft_r2c(plan, const_cast< T * >(&h(0, 0)),
      reinterpret_cast< T (*)[2] >(&hf(0, 0)));
    hf /= NR * NC;
  }

  ~conv_fft() {
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(inv_plan);
  }

  void operator()(dlib::matrix< T, NR, NC > &x) const {
    dlib::matrix< std::complex< T >, NR, NC_Comp > xf;

    fftw_helper< T >::execute_dft_r2c(plan, const_cast< T * >(&x(0, 0)),
      reinterpret_cast< T (*)[2] >(&xf(0, 0)));
    xf = pointwise_multiply(xf, hf);

    fftw_helper< T >::execute_dft_c2r(inv_plan,
      reinterpret_cast< T (*)[2] >(&xf(0, 0)), &x(0, 0));
  }

private:
  static const long NC_Comp = NC / 2 + 1;

  typename fftw_helper< T >::plan_type plan, inv_plan;

  dlib::matrix< std::complex< T >, NR, NC_Comp > hf;
};

#endif
