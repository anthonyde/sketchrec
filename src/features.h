#ifndef FEATURES_H
#define FEATURES_H

#include <cassert>
#include <cmath>

#include <dlib/matrix.h>

#include "conv.h"
#include "util.h"

// Bin the gradient magnitudes by orientation into orientational response
// images.
template< class T, long NR, long NC >
void orient_responses(const dlib::matrix< T, NR, NC > &g,
  const dlib::matrix< T, NR, NC > &o, unsigned int bin_count,
  std::vector< dlib::matrix< T, NR, NC > > &os) {
  const T w = M_PI / bin_count; // The width of each bin
  os.assign(bin_count, zeros_matrix(g));

  for (unsigned int k = 0; k < bin_count; ++k) {
    const T o0 = (k - .5) * w; // The lower bound for this bin
    const T oc = k * w; // The center of this bin
    const T o1 = (k + .5) * w; // The upper bound for this bin

    T oji;
    for (long j = 0; j < NR; ++j) {
      for (long i = 0; i < NC; ++i) {
        oji = o(j, i);
        if (oji >= o0 && oji < o1) {
          const T t = std::abs(oc - oji) / w;
          os[k](j, i) += g(j, i) * (1. - t);
          os[((oji < oc) ? k + bin_count - 1 : k + 1) % bin_count](j, i) +=
            g(j, i) * t;
        }
      }
    }
  }
}

// A class for extracting feature descriptors from a grayscale image
template< class T, long N >
struct feature_desc_extractor {
  static const unsigned int orient_bin_count = 4;
  static const unsigned int spatial_bin_count = 4; // In each dimension
  static const unsigned int spatial_bin_size =
    (N * 0.35355339) / spatial_bin_count; // 12.5% area
  static const unsigned int feature_grid_size = 28;

  typedef dlib::matrix< T, N, N > image_type;
  typedef dlib::matrix< T, orient_bin_count *
    spatial_bin_count * spatial_bin_count, 1 > desc_type;

  static void extract(const image_type &image,
    std::vector< desc_type > &descs) {
    descs.clear();
    descs.reserve(feature_grid_size * feature_grid_size);

    // Compute the gradient.
    const image_type gx = conv_same(image, sobel_x);
    const image_type gy = conv_same(image, sobel_y);

    // Compute the magnitude and orientation of the gradient.
    image_type g, o;
    cart2polar(gx, gy, g, o);

    // Limit the orientation range to [0, pi).
    for (long j = 0; j < N; ++j) {
      for (long i = 0; i < N; ++i) {
        if (o(j, i) >= M_PI)
          o(j, i) = M_PI - o(j, i);
      }
    }

    // Generate orientational response images.
    std::vector< image_type > os;
    orient_responses(g, o, orient_bin_count, os);

    // Convolve each orientational response image with a 2D tent function to
    // accelerate interpolation.
    for (unsigned int i = 0; i < orient_bin_count; ++i) {
      conv_tent(os[i]);
      // Account for slightly negative responses introduced by the FFT.
      os[i] = abs(os[i]);
    }

    // Extract feature descriptors on a regular grid.  Orientational response
    // values are binned into a spatial grid centered at each grid point.
    static const unsigned int dg = N / feature_grid_size;

    for (unsigned int v = dg / 2; v < N; v += dg) {
      for (unsigned int u = dg / 2; u < N; u += dg) {
        desc_type d;

        for (unsigned int i = 0; i < orient_bin_count; ++i) {
          for (unsigned int t = 0; t < spatial_bin_count; ++t) {
            // The vertical bin center
            const int ct = spatial_bin_size / 2 + spatial_bin_size *
              (t - spatial_bin_count / 2);

            for (unsigned int s = 0; s < spatial_bin_count; ++s) {
              // The horizontal bin center
              const int cs = spatial_bin_size / 2 + spatial_bin_size *
                (s - spatial_bin_count / 2);

              const int y = v + ct;
              const int x = u + cs;
              d((i * spatial_bin_count + t) * spatial_bin_count + s) =
                (0 <= y && y < N && 0 <= x && x < N) ? os[i](y, x) : 0;
            }
          }
        }

        // Normalize the feature descriptor before adding it to the array.
        descs.push_back(normalize(d));
      }
    }
  }

private:
  // A 2D tent function kernel for bilinear interpolation
  static image_type tent_kernel_init() {
    const unsigned int tent_size = 2 * spatial_bin_size + 1;

    image_type m;
    m = 0;
    for (unsigned int j = 0; j < tent_size; ++j) {
      const unsigned int xj =
        spatial_bin_size - std::abs(j - spatial_bin_size);
      for (unsigned int i = 0; i < tent_size; ++i)
        m(j, i) = xj * (spatial_bin_size - std::abs(i - spatial_bin_size));
    }
    return m;
  }

  static const conv_fft< T, N, N > conv_tent;
};

template< class T, long N >
const conv_fft< T, N, N > feature_desc_extractor< T, N >::conv_tent(
  feature_desc_extractor::tent_kernel_init());

// A helper function for inferring the image type for feature extraction
template< class T, long N >
void extract_descriptors(const dlib::matrix< T, N, N > &S,
  std::vector< typename feature_desc_extractor< T, N >::desc_type > &D) {
  feature_desc_extractor< T, N >::extract(S, D);
}

// Quantize a feature descriptor for a vocabulary using the Gaussian distance
// to each word.
template< class T, long N, long V >
void quantize_desc(const dlib::matrix< T, N, 1 > &desc,
  const std::vector< dlib::matrix< T, N, 1 > > &vocab,
  dlib::matrix< T, V, 1 > &q) {
  static const T sigma = .1; // Sigma for Gaussian distance
  typedef std::vector< dlib::matrix< T, N, 1 > > vocab_type;

  assert(vocab.size() == V);

  for (typename vocab_type::size_type i = 0; i < vocab.size(); ++i) {
    const dlib::matrix< T, N, 1 > diff = desc - vocab[i];
    q(i) = std::exp(-dot(diff, diff) / (2 * sigma * sigma));
  }
}

// Generate a feature histogram for a set of feature descriptors and a
// vocabulary.
template< class T, long N, long V >
void feature_hist(const std::vector< dlib::matrix< T, N, 1 > > &descs,
  const std::vector< dlib::matrix< T, N, 1 > > &vocab,
  dlib::matrix< T, V, 1 > &hist) {
  assert(vocab.size() == V && V > 0);

  hist = 0;

  for (const auto &desc : descs) {
    dlib::matrix< T, V, 1 > q;
    quantize_desc(desc, vocab, q);

    // Normalize the feature distance before accumulating.
    hist += l1_normalize(q);
  }

  hist /= V;
}

#endif
