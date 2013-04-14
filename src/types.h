#ifndef TYPES_H
#define TYPES_H

#include <vector>

#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/type_safe_union.h>

#include "features.h"
#include "svm.h"

// Preprocessing
typedef feature_desc_extractor< float, 256 > feature_desc_extractor_type;
typedef feature_desc_extractor_type::image_type image_type;
typedef feature_desc_extractor_type::desc_type feature_desc_type;
typedef std::vector< feature_desc_type > vocab_type;
typedef dlib::matrix< float, 500, 1 > feature_hist_type;

// Classification
typedef dlib::radial_basis_kernel< feature_hist_type > kernel_type;
typedef dlib::svm_c_trainer< kernel_type > trainer_type;

typedef one_vs_all_trainer2< dlib::any_trainer< feature_hist_type, float >,
  int, true > ova_trainer_type;
typedef dlib::one_vs_all_decision_function< ova_trainer_type,
  dlib::decision_function< kernel_type > > ova_df_type;

typedef one_vs_one_trainer2< dlib::any_trainer< feature_hist_type, float >,
  int, true > ovo_trainer_type;
typedef dlib::one_vs_one_decision_function< ovo_trainer_type,
  dlib::decision_function< kernel_type > > ovo_df_type;

typedef dlib::type_safe_union< ova_df_type, ovo_df_type > df_type;

#endif
