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



class GlobalTrainingExample;

typedef std::vector<std::shared_ptr<GlobalExample>> GlobalTrainingExamplePtrs;

/**
 * example for greedy training
 */
class Example {

public:
    FeatureVectors feature_vectors;
    std::vector<int> predict_labels;

    Example(const FeatureVectors& fs, const std::vector<int>& l) : feature_vectors(fs), predict_labels(l){
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

    GlobalExample(std::vector<Example> &exs, std::vector<int> &gActs, Instance &inst): examples(exs), gold_actions(gActs), instance(inst) {}
    ~GlobalExample() {}
};

#endif //SNNOW_TRAININGEXAMPLE_H
