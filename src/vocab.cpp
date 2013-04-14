#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <dlib/matrix.h>

#include "features.h"
#include "io.h"
#include "kmeans.h"
#include "svg.h"
#include "types.h"
#include "util.h"

int main(int argc, char *argv[]) {
  typedef stream_sample< feature_desc_type > stream_sample_type;

  // Process the command-line arguments.
  typename stream_sample_type::size_type n = 1000000;
  const char *vocab_path = "vocab.out";

  {
    int i;
    for (i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "-h")) {
        goto usage;
      }
      else if (!strcmp(argv[i], "-n")) {
        std::istringstream ss(argv[++i]);
        if (!(ss >> n))
        goto usage;
        }
      else {
        break;
      }
    }

    if (i < argc)
      vocab_path = argv[i++];

    if (i != argc)
      goto usage;
  }

  {
    // Extract features for all input files.
    std::vector< std::string > paths;
    std::string path;
    while (std::getline(std::cin, path))
      paths.push_back(path);

    // Select a fixed number of random descriptors.
    std::random_device rd;
    std::mt19937 gen(rd());
    stream_sample_type samples(n);

    #pragma omp parallel for schedule(dynamic)
    for (typename std::vector< std::string >::size_type i = 0;
      i < paths.size(); ++i) {
      const std::string &path = paths[i];

      #pragma omp critical
      {
        std::cout << "Extracting features for " << path << " (" << i + 1
          << '/' << paths.size() << ")...\n";
      }

      image_type image;
      load_svg(path.c_str(), image);
      image = 1. - image;

      std::vector< feature_desc_type > descs;
      extract_descriptors(image, descs);

      #pragma omp critical
      {
        for (const auto &desc : descs)
          samples.push_back(gen, desc);
      }
    }

    std::cout << "Got " << samples.get().size() << " descriptors\n";

    // Generate a vocabulary for this data set.
    std::cout << "Clustering...\n";

    static const long center_count = feature_hist_type::NR;
    std::cout << "Picking " << center_count << " initial centers...\n";
    vocab_type vocab;
    kmeanspp< float >(gen, samples.get(), center_count, vocab);

    kmeans< float, feature_desc_type, true >(samples.get(), vocab);

    // Save the vocabulary.
    std::cout << "Saving vocabulary...\n";
    {
      std::ofstream fs(vocab_path, std::ios::binary);
      serialize2(vocab, fs);
    }
  }

  return 0;

usage:
  std::cerr << "Usage: " << argv[0] << " [-n sample-count] [vocab-file]\n";
  return 1;
}

