#ifndef KMEANS_H
#define KMEANS_H

#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <dlib/matrix.h>

// An implementation of the k-means++ cluster center initialization algorithm
// by Arthur and Vassilvitskii
template< class DistT, class T, class Generator >
void kmeanspp(Generator &g, const std::vector< T > &samples,
  typename std::vector< T >::size_type k, std::vector< T > &centers) {
  typedef std::vector< T > vector_type;

  assert(samples.size() > 0 && k > 0);

  // Initialize storage for the centers and the minimum distance to a center
  // from each sample point.
  std::vector< DistT > min_distances(samples.size(),
    std::numeric_limits< DistT >::max());
  centers.clear();
  centers.reserve(k);

  // Pick the first center uniformly at random.
  {
    std::uniform_int_distribution< typename vector_type::size_type >
      uniform_sample_index(0, samples.size() - 1);
    centers.push_back(samples[uniform_sample_index(g)]);
  }

  // Pick the remaining centers.
  for (typename vector_type::size_type i = 0; i < k - 1; ++i) {
    // Update the minimum distance from each sample to a center, taking into
    // account them most recently added center.
    #pragma omp parallel for
    for (typename vector_type::size_type j = 0; j < samples.size(); ++j) {
      const DistT dist = dlib::length_squared(centers[i] - samples[j]);
      if (dist < min_distances[j])
        min_distances[j] = dist;
    }

    // Pick the next center at random using a probability distribution
    // weighted by distance squared.
    std::discrete_distribution< typename vector_type::size_type >
      weighted_sample_index(min_distances.begin(), min_distances.end());
    centers.push_back(samples[weighted_sample_index(g)]);
  }
}

// An implementation of k-means clustering
template< class DistT, class T, bool Verbose = false >
void kmeans(const std::vector< T > &samples, std::vector< T > &centers,
  unsigned int max_iter = 1000) {
  typedef std::vector< T > vector_type;

  assert(samples.size() > 0 && centers.size() > 0);

  // A zero sample for calculating the centroid
  T zero = dlib::zeros_matrix(centers[0]);

  // Initialize storage for the number of samples for each center and the
  // center associated with each sample.
  std::vector< typename vector_type::size_type > assignments(samples.size());
  std::vector< typename vector_type::size_type > center_element_count;

  if (Verbose) {
    std::cout << "Running k-means...";
    std::cout.flush();
  }

  unsigned int iter = 0;
  bool centers_changed = true;
  while (centers_changed && iter < max_iter) {
    ++iter;
    centers_changed = false;

    if (Verbose) {
      std::cout << ' ' << iter << "...";
      std::cout.flush();
    }

    // Determine which center each sample is closest to.
    #pragma omp parallel for
    for (typename vector_type::size_type i = 0; i < samples.size(); ++i) {
      DistT min_dist = std::numeric_limits< DistT >::max();
      typename vector_type::size_type min_center = 0;

      for (typename vector_type::size_type j = 0; j < centers.size(); ++j) {
        const DistT dist = dlib::length_squared(centers[j] - samples[i]);
        if (dist < min_dist) {
          min_dist = dist;
          min_center = j;
        }
      }

      if (assignments[i] != min_center) {
        centers_changed = true;
        assignments[i] = min_center;
      }
    }

    // Update the cluster centers.
    centers.assign(centers.size(), zero);
    center_element_count.assign(centers.size(), 0);

    for (typename vector_type::size_type i = 0; i < samples.size(); ++i) {
      const typename vector_type::size_type &assignment = assignments[i];
      centers[assignment] += samples[i];
      ++center_element_count[assignment];
    }

    for (typename vector_type::size_type i = 0; i < centers.size(); ++i) {
      if (center_element_count[i])
        centers[i] /= center_element_count[i];
    }
  }

  if (Verbose)
    std::cout << " done\n";
}

#endif
