/*************************************************************************
	> File Name: SeqLabeler.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:29:06 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELER_H
#define SNNOW_SEQLABELER_H

#include <memory>
#include <gflags/gflags.h>
#include <chrono>
#include <algorithm>
#include <fstream>

#include "base/NLPCore.h"
#include "DataSet.h"
#include "ChunkerFeatureExtractor.h"
#include "ChunkerTransitionSystem.h"

// DECLARE_string(embedding_file);
// DECLARE_string(training_file);
// DECLARE_string(test_file);
// DECLARE_string(dev_file);
// DECLARE_string(model_file);

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

class SeqLabeler : public NLPCore {
private:
    bool b_train_;
    std::shared_ptr<ChunkerFeatureExtractor> feature_extractor_ptr_;
    std::shared_ptr<ChunkerTransitionSystem> transition_system_ptr_;

public:
    SeqLabeler() = default;

    SeqLabeler(bool b_train);

    ~SeqLabeler() = default;

    // train the input sentences with mini-batch adaGrad
    void train(DataSet& training_set, DataSet& dev_set);

    void greedyTrain(DataSet& training_set, DataSet& dev_set);

    double test(DataSet &test_data, Model<cpu> & model, FeedForwardNNet<gpu> & net);

    void trainInit(DataSet& training_set);

private:
};

#endif //SNNOW_SEQLABELER_H
