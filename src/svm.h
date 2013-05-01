#ifndef SVM_H
#define SVM_H

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/unordered_pair.h>

// A trainer for one-vs-all multi-class classifiers
template< class AnyTrainer, class LabelT, bool Verbose = false >
struct one_vs_all_trainer2 {
  typedef LabelT label_type;
  typedef typename AnyTrainer::sample_type sample_type;
  typedef typename AnyTrainer::scalar_type scalar_type;
  typedef typename AnyTrainer::mem_manager_type mem_manager_type;
  typedef dlib::one_vs_all_decision_function< one_vs_all_trainer2 >
    trained_function_type;

  one_vs_all_trainer2(const AnyTrainer &trainer_) : trainer(trainer_) {
  }

  trained_function_type train(const std::vector< sample_type > &samples,
    const std::vector< label_type > &labels) const {
    assert(dlib::is_learning_problem(samples, labels));

    const std::vector< label_type > distinct_labels =
      dlib::select_all_distinct_labels(labels);
    typename trained_function_type::binary_function_table dfs;

    #pragma omp parallel
    {
      std::vector< scalar_type > set_labels;

      #pragma omp for schedule(dynamic)
      for (typename std::vector< label_type >::size_type i = 0;
        i < distinct_labels.size(); ++i) {
        const label_type &label = distinct_labels[i];
        set_labels.clear();

        // Set up the one-vs-all training set.
        for (typename std::vector< sample_type >::size_type k = 0;
          k < samples.size(); ++k)
          set_labels.push_back((labels[k] == label) ? 1 : -1);

        if (Verbose) {
          #pragma omp critical
          {
            std::cout << "Training classifier " << i + 1 << '/'
              << distinct_labels.size() << "...\n";
          }
        }

        // Train the classifier.
        typename trained_function_type::binary_function_table::mapped_type
          df = trainer.train(samples, set_labels);

        #pragma omp critical
        {
          dfs[label] = df;
        }
      }
    }

    return trained_function_type(dfs);
  }

private:
  AnyTrainer trainer;
};

// A trainer for one-vs-one multi-class classifiers
template< class AnyTrainer, class LabelT, bool Verbose = false >
struct one_vs_one_trainer2 {
  typedef LabelT label_type;
  typedef typename AnyTrainer::sample_type sample_type;
  typedef typename AnyTrainer::scalar_type scalar_type;
  typedef typename AnyTrainer::mem_manager_type mem_manager_type;
  typedef dlib::one_vs_one_decision_function< one_vs_one_trainer2 >
    trained_function_type;

  one_vs_one_trainer2(const AnyTrainer &trainer_) : trainer(trainer_) {
  }

  trained_function_type train(const std::vector< sample_type > &samples,
    const std::vector< label_type > &labels) const {
    assert(dlib::is_learning_problem(samples, labels));

    const std::vector< label_type > distinct_labels =
      dlib::select_all_distinct_labels(labels);
    typename trained_function_type::binary_function_table dfs;

    typename std::vector< label_type >::size_type n = 0;
    #pragma omp parallel
    {
      std::vector< sample_type > set_samples;
      std::vector< scalar_type > set_labels;

      #pragma omp for schedule(dynamic)
      for (typename std::vector< label_type >::size_type i = 0;
        i < distinct_labels.size(); ++i) {
        const label_type &label1 = distinct_labels[i];
        for (typename std::vector< label_type >::size_type j = i + 1;
          j < distinct_labels.size(); ++j) {
          const label_type &label2 = distinct_labels[j];
          const dlib::unordered_pair< label_type > pair(label1, label2);

          // Set up the one-vs-one training set.
          set_samples.clear();
          set_labels.clear();
          for (typename std::vector< sample_type >::size_type k = 0;
            k < samples.size(); ++k) {
            if (labels[k] == pair.first) {
              set_samples.push_back(samples[k]);
              set_labels.push_back(1);
            }
            else if (labels[k] == pair.second) {
              set_samples.push_back(samples[k]);
              set_labels.push_back(-1);
            }
          }

          if (Verbose) {
            #pragma omp critical
            {
              std::cout << "Training classifier " << n + 1 << '/'
                << distinct_labels.size() * (distinct_labels.size() - 1) / 2
                << "...\n";
              ++n;
            }
          }

          // Train the classifier.
          typename trained_function_type::binary_function_table::mapped_type
            df = trainer.train(set_samples, set_labels);

          #pragma omp critical
          {
            dfs[pair] = df;
          }
        }
      }
    }

    return trained_function_type(dfs);
  }

private:
  AnyTrainer trainer;
};

// Run a multi-class decision function on a test set, returning the confusion
// matrix.
template< class DF, class SampleT, class LabelT, bool Verbose = false >
const dlib::matrix< double > test_multiclass_decision_function2(
  const DF &df, const std::vector< SampleT > &test_samples,
  const std::vector< LabelT > &test_labels) {
  typedef std::map< LabelT, typename std::vector< LabelT >::size_type >
    label_count_map_type;
  typedef typename DF::mem_manager_type mem_manager_type;

  assert(is_learning_problem(test_samples, test_labels));

  const std::vector< LabelT > &labels = df.get_labels();

  label_count_map_type label_offsets;
  for (typename std::vector< LabelT >::size_type i = 0;
    i < labels.size(); ++i)
    label_offsets[labels[i]] = i;

  dlib::matrix< double, 0, 0, mem_manager_type > conf(labels.size(),
    labels.size());
  conf = 0;

  typename std::vector< LabelT >::size_type n = 0;
  #pragma omp parallel for
  for (typename std::vector< SampleT >::size_type i = 0;
    i < test_samples.size(); ++i) {
    const auto it = label_offsets.find(test_labels[i]);
    assert(it != label_offsets.end());

    if (Verbose) {
      #pragma omp critical
      {
        std::cout << "Classifying sample " << n + 1 << '/'
          << test_samples.size() << "...\n";
        ++n;
      }
    }

    const auto pred_offset = label_offsets.find(df(test_samples[i]))->second;

    #pragma omp critical
    {
      ++conf(it->second, pred_offset);
    }
  }

  return conf;
}

// Cross-validation for multi-class classifiers
template< class Trainer, class SampleT, class LabelT, bool Verbose = false >
const dlib::matrix< double > cross_validate_multiclass_trainer2(
  const Trainer &trainer, const std::vector< SampleT > &samples,
  const std::vector< LabelT > &labels, const unsigned long folds) {
  typedef std::map< LabelT, typename std::vector< LabelT >::size_type >
    label_count_map_type;
  typedef typename Trainer::mem_manager_type mem_manager_type;

  assert(is_learning_problem(samples, labels) && 1 < folds &&
    folds <= samples.size());

  const std::vector< LabelT > distinct_labels =
    dlib::select_all_distinct_labels(labels);

  // Count the occurrences of each label.
  label_count_map_type label_counts;
  for (const auto &label : labels)
    ++label_counts[label];

  // Determine the sizes of the test and the training sets.
  label_count_map_type test_sizes, train_sizes;
  for (const auto &pair : label_counts) {
    const typename label_count_map_type::mapped_type test_size =
      pair.second / folds;
    if (!test_size) {
      std::ostringstream ss;
      ss << "In cross_validate_multiclass_trainer2(), the number of folds"
        " was larger than the number of elements in one of the training"
        " classes.\n  folds: " << folds << "\n  size of class: "
        << pair.second << '\n';
      throw dlib::cross_validation_error(ss.str());
    }

    test_sizes[pair.first] = test_size;
    train_sizes[pair.first] = pair.second - test_size;
  }

  dlib::matrix< double, 0, 0, mem_manager_type > conf(labels.size(),
    labels.size());
  conf = 0;

  label_count_map_type next_offsets;

  std::vector< SampleT > test_samples, train_samples;
  std::vector< LabelT > test_labels, train_labels;

  // Train and test with each fold configuration.
  for (unsigned long i = 0; i < folds; ++i) {
    test_samples.clear();
    train_samples.clear();
    test_labels.clear();
    train_labels.clear();

    // Load the test samples.
    for (const auto &label : distinct_labels) {
      const auto test_size = test_sizes[label];

      unsigned long &next_offset = next_offsets[label];
      unsigned long size = 0;
      while (size < test_size) {
        if (labels[next_offset] == label) {
          test_samples.push_back(samples[next_offset]);
          test_labels.push_back(label);
          ++size;
        }

        next_offset = (next_offset + 1) % samples.size();
      }
    }

    // Load the training samples.
    for (const auto &label : distinct_labels) {
      const auto train_size = train_sizes[label];

      unsigned long &next_offset = next_offsets[label];
      unsigned long size = 0;
      while (size < train_size) {
        if (labels[next_offset] == label) {
          train_samples.push_back(samples[next_offset]);
          train_labels.push_back(label);
          ++size;
        }

        next_offset = (next_offset + 1) % samples.size();
      }
    }

    if (Verbose) {
      std::cout << "Running cross-validation on fold " << i + 1 << '/'
        << folds << "...\n";
    }

    conf += test_multiclass_decision_function2<
      typename Trainer::trained_function_type, SampleT, LabelT, Verbose >(
      trainer.train(train_samples, train_labels), test_samples, test_labels);
  }

  return conf;
}

#endif
