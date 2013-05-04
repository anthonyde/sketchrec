#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "features.h"
#include "io.h"
#include "svm.h"
#include "svg.h"
#include "types.h"

int main(int argc, char *argv[]) {
  // Process the command-line arguments.
  const char *vocab_path = "vocab.out";
  const char *map_path = "map_id_label.txt";
  const char *cats_path = "cats.out";
  bool ova = true;
  typename kernel_type::scalar_type gamma = 17.8;
  typename kernel_type::scalar_type c = 3.2;

  {
    int i;
    for (i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "-h")) {
        goto usage;
      }
      else if (!strcmp(argv[i], "-v")) {
        vocab_path = argv[++i];
      }
      else if (!strcmp(argv[i], "-m")) {
        map_path = argv[++i];
      }
      else if (!strcmp(argv[i], "-c")) {
        ++i;
        if (!strcmp(argv[i], "ova")) {
          ova = true;
        }
        else if (!strcmp(argv[i], "ovo")) {
          ova = false;
        }
        else {
          std::cerr << argv[0] << ": Unsupported classifier: `" << argv[i]
            << "'\n";
          goto err;
        }
      }
      else if (!strcmp(argv[i], "-g")) {
        std::istringstream ss(argv[++i]);
        if (!(ss >> gamma))
          goto usage;
      }
      else if (!strcmp(argv[i], "-C")) {
        std::istringstream ss(argv[++i]);
        if (!(ss >> c))
          goto usage;
      }
      else {
        break;
      }
    }

    if (i < argc)
      cats_path = argv[i++];

    if (i != argc)
      goto usage;

    if (!vocab_path || !map_path)
      goto usage;
  }

  {
    // Load the vocabulary.
    std::cout << "Loading vocabulary...\n";
    vocab_type vocab;
    {
      std::ifstream fs(vocab_path, std::ios::binary);
      deserialize2(vocab, fs);
    }

    // Load the category map.
    std::cout << "Loading category map...\n";
    std::map< std::string, int > cat_map;
    {
      std::ifstream fs(map_path);
      for (std::string line; std::getline(fs, line);) {
        std::istringstream ss(line);
        int i;
        std::string label;
        ss >> i;
        ss.get(); // ','
        std::getline(ss, label);
        cat_map[label] = i;
      }
      fs.close();
    }

    // Extract features for all input files.
    std::vector< feature_hist_type > samples;
    std::vector< int > labels;

    std::vector< std::string > paths;
    std::string path;
    while (std::getline(std::cin, path))
      paths.push_back(path);

    #pragma omp parallel for schedule(dynamic)
    for (typename std::vector< std::string >::size_type i = 0;
      i < paths.size(); ++i) {
      const std::string &path = paths[i];

      #pragma omp critical
      {
        std::cout << "Extracting features for " << path << " (" << i + 1
          << '/' << paths.size() << ")...\n";
      }

      // Get the category from the directory name.
      const std::size_t dir_end = path.rfind('/');
      std::size_t dir_begin = path.rfind('/', dir_end - 1);
      if (dir_begin == std::string::npos)
        dir_begin = 0;
      else
        ++dir_begin;

      const std::string dir = path.substr(dir_begin, dir_end - dir_begin);
      const int cat = cat_map[dir];

      assert(cat);

      // Extract the features.
      image_type image;
      load_svg(path.c_str(), image);
      image = 1. - image;

      std::vector< feature_desc_type > descs;
      extract_descriptors(image, descs);

      feature_hist_type hist;
      feature_hist(descs, vocab, hist);

      // Store the category label and feature histogram.
      #pragma omp critical
      {
        samples.push_back(hist);
        labels.push_back(cat);
      }
    }

    // Train a multi-class classifier.
    trainer_type rbf_trainer;
    rbf_trainer.set_kernel(kernel_type(gamma));
    rbf_trainer.set_c(c);

    df_type df;
    if (ova) {
      std::cout << "Training one-vs-all classifier...\n";
      df.get< ova_df_type >() =
        ova_trainer_type(rbf_trainer).train(samples, labels);
    }
    else {
      std::cout << "Training one-vs-one classifier...\n";
      df.get< ovo_df_type >() =
        ovo_trainer_type(rbf_trainer).train(samples, labels);
    }

    // Save the classifier.
    std::cout << "Saving classifier...\n";
    {
      std::ofstream fs(cats_path, std::ios::binary);
      if (ova)
        serialize2(df.get< ova_df_type >(), fs);
      else
        serialize2(df.get< ovo_df_type >(), fs);
    }
  }

  return 0;

usage:
  std::cerr << "Usage: " << argv[0] << " [-v vocab-file] [-m map-file]"
    " [-c classifier] [-g gamma] [-C C] [cats-file]\n";
err:
  return 1;
}

