/*************************************************************************
    > File Name: FeatureManager.h
    > Author: cheng chuan
    > Mail: cc.square0@gmail.com 
    > Created Time: Thu 24 Dec 2015 04:17:50 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREMANAGER_H_
#define _CHUNKER_COMMON_FEATUREMANAGER_H_

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <tr1/unordered_map>

#include "mshadow/tensor.h"

#include "DictManager.h"
#include "FeatureExtractor.h"
#include "FeatureVector.h"
#include "LabeledSequence.h"
#include "Instance.h"
#include "State.h"

class FeatureManager {
private:
    std::vector<std::shared_ptr<FeatureExtractor>> featExtractorPtrs;

public:
    FeatureManager() { }
    ~FeatureManager() { }

    void init(const ChunkedDataSet &goldSet, const std::shared_ptr<DictManager> &dataManagerPtr);

    void extractFeature(const State &state, const Instance &inst, FeatureVector &features);

    std::vector<FeatureType> getFeatureTypes();

    std::vector<std::shared_ptr<Dictionary>> getDictManagerPtrs();

private:
    FeatureManager(const FeatureManager &fManager) = delete;
    FeatureManager& operator= (const FeatureManager &fManager) = delete;
};

#endif
