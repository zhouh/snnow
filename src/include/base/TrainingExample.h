//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_TRAININGEXAMPLE_H
#define SNNOW_TRAININGEXAMPLE_H

#include <vector>
#include <iostream>
#include <memory>

#include "base/Input.h"
#include "FeatureVector.h"


/**
 * example for greedy training
 */
class Example {

public:
    FeatureVector feature_vector;
    int predict_label;

    Example(const FeatureVector& fs, int l) : feature_vector(fs), predict_label(l){
    }

    ~Example() {}
};

/**
 * training example for beam search
 */
class GlobalExample {
public:
    std::vector<Example> examples;
    std::vector<int> gold_actions;
    Input input;

    GlobalExample(std::vector<Example> &exs, std::vector<int> &gActs): examples(exs), gold_actions(gActs){}
    ~GlobalExample() {}
};


typedef std::vector<std::shared_ptr<GlobalExample>> GlobalTrainingExamplePtrs;

#endif //SNNOW_TRAININGEXAMPLE_H
