#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "features.h"
#include "io.h"
#include "svg.h"
#include "svm.h"
#include "types.h"

int main(int argc, char *argv[]) {
  // Process the command-line arguments.
  const char *vocab_path = "vocab.out";
  const char *map_path = "map_id_label.txt";
  const char *cats_path = "cats.out";
  bool ova = true;

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
      else {
        break;
      }
    }

    if (i < argc)
      cats_path = argv[i++];

    if (i != argc)
      goto usage;

    if (!vocab_path || !map_path || !cats_path)
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
    std::map< int, std::string > cat_map;
    {
      std::ifstream fs(map_path);
      for (std::string line; std::getline(fs, line);) {
        std::istringstream ss(line);
        int i;
        std::string label;
        ss >> i;
        ss.get(); // ','
        std::getline(ss, label);
        cat_map[i] = label;
      }
    }

    // Load the category classifier.
    std::cout << "Loading classifier...\n";
    df_type df;
    {
      std::ifstream fs(cats_path, std::ios::binary);
      if (ova)
        deserialize2(df.get< ova_df_type >(), fs);
      else
        deserialize2(df.get< ovo_df_type >(), fs);
    }

    // Extract features for all input files.
    std::vector< std::string > paths;
    std::string path;
    while (std::getline(std::cin, path))
      paths.push_back(path);

    #pragma omp parallel for schedule(dynamic)
    for (typename std::vector< std::string >::size_type i = 0;
      i < paths.size(); ++i) {
      const std::string &path = paths[i];

      // Extract the features.
      image_type image;
      load_svg(path.c_str(), image);
      image = 1. - image;

      std::vector< feature_desc_type > descs;
      extract_descriptors(image, descs);

      feature_hist_type hist;
      feature_hist(descs, vocab, hist);

      const int cat = ova ? df.get< ova_df_type >()(hist) :
        df.get< ovo_df_type >()(hist);

      assert(cat);

      #pragma omp critical
      {
        std::cout << path << ' ' << cat_map[cat] << '\n';
      }
    }
  }

  return 0;

usage:
  std::cerr << "Usage: " << argv[0]
    << " [-v vocab-file] [-m map-file] [-c classifier] [cats-file]\n";
err:
  return 1;
}

