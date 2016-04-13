//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DepParser_H
#define SNNOW_DepParser_H

#include <memory>
#include <gflags/gflags.h>

#include "base/NLPCore.h"
#include "DataSet.h"
#include "DepArcStandardSystem.h"
#include "DepParseFeatureExtractor.h"

DECLARE_string(word_embedding_dim);
DECLARE_int32(beam_size);

class DepParser : public NLPCore {

private:

    DepParseFeatureExtractor feature_extractor;
    std::shared_ptr<FeatureEmbedding> feature_embedding_handler_ptr;
    std::shared_ptr<DepArcStandardSystem> trainsition_system_ptr;

    int beam_size;
    bool be_train;
    bool be_early_update = true;

public:
    DepParser(bool bTrain);
    DepParser();
    ~DepParser();

    // train the input sentences with mini-batch adaGrad
    void train(DataSet& training_set, DataSet& dev_set);

    void greedyTrain(DataSet& training_set, DataSet& dev_set);

    double test(DataSet& test_set);

    void trainInit(DataSet& training_set);


};


#endif //SNNOW_DepParser_H
