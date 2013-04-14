#ifndef SVM_H
#define SVM_H

#include <cassert>
#include <iostream>
#include <vector>

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

#endif
