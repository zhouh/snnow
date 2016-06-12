//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DepParser_H
#define SNNOW_DepParser_H

#include <memory>
#include <gflags/gflags.h>
#include <chrono>
#include <algorithm>
#include <fstream>


#include "base/NLPCore.h"
#include "DataSet.h"
#include "DepParseMacro.h"
#include "DepArcStandardSystem.h"
#include "DepParseFeatureExtractor.h"
#include "FeatureType.h"
#include "nets/Model.h"
#include "nets/FeedForwardNNet.h"
#include "DepParseEvalb.h"



DECLARE_string(embedding_file);
DECLARE_string(training_file);
DECLARE_string(test_file);
DECLARE_string(dev_file);
DECLARE_string(model_file);

DECLARE_int32(max_training_iteration_num);
DECLARE_int32(batch_size);
DECLARE_int32(thread_num);
DECLARE_int32(word_embedding_dim);
DECLARE_int32(label_num);
DECLARE_int32(beam_size);
DECLARE_int32(hidden_size);
DECLARE_int32(feature_num);
DECLARE_int32(evaluate_per_iteration);

DECLARE_bool(be_dropout);

DECLARE_double(learning_rate);
DECLARE_double(init_range);
DECLARE_double(regularization_rate);
DECLARE_double(dropout_prob);
DECLARE_double(adagrad_eps);

class DepParser : public NLPCore {

private:

    std::shared_ptr<DepParseFeatureExtractor> feature_extractor_ptr;
    std::shared_ptr<DepArcStandardSystem> trainsition_system_ptr;

    // examples for greedy training
    std::vector<std::shared_ptr<Example>> greedy_example_ptrs;

    int beam_size;
    bool be_train;
    bool be_early_update = true;

public:
    DepParser(bool bTrain);
    DepParser() = default;
    ~DepParser() = default;

    // train the input sentences with mini-batch adaGrad
    void train(DataSet& training_set, DataSet& dev_set);

    void greedyTrain(DataSet& training_set, DataSet& dev_set);

    double test(DataSet &test_data, Model<cpu> & model, FeedForwardNNet<gpu> & net);

    void trainInit(DataSet& training_set);


};


#endif //SNNOW_DepParser_H
