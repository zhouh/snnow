//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_NLPCORE_H
#define SNNOW_NLPCORE_H

#include <vector>

#include "DataSet.h"

/**
 * The base object for parser, segmentor tagger and chunker.
 */
class NLPCore {

public:
    virtual void train(DataSet& training_set, DataSet& dev_set) = 0;

    /**
     * test function
     * return the evaluation score
     */
    virtual double test(DataSet& test_set) = 0;
};


#endif //SNNOW_NLPCORE_H
