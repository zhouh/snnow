/*************************************************************************
	> File Name: FeatureManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 04:17:50 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREMANAGER_H_
#define _CHUNKER_FEATUREMANAGER_H_

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <pair>
#include <memory>
#include <tr1/unordered_map>

#include "mshadow/tensor.h"

#include "FeatureExtractor.h"
#include "DataManager.h"

#include "FeatureVector.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"
#include "State.h"

class FeatureManager {
public:
    std::vector<std::shared_ptr<FeatureExtractor>> featExtractorPtrs;
    std::vector<std::pair<bool, std::string>> isReadFeatPretrainedEmbs;
    int totalFeatSize;

public:
    FeatureManager() { }
    ~FeatureManager() { }

    void init(const ChunkedDataSet &goldSet, const std::shared_ptr<DataManager> &dataManagerPtr);

    void extractFeature(State &state, Instance &inst, FeatureVector &features);

    std::vector<FeatureType> getFeatureTypes();

    std::vector<std::shared_ptr<DictManager>> getDictManagerPtrs();

    std::vector<std::pair<bool, std::string>> getIsReadPretrainedEmbs() {
        return isReadFeatPretrainedEmbs;
    }

    /*
     * construct the input by x = beamIndex, y = featureLayerIndex
     */
	void returnInput(std::vector<FeatureVector> &featVecs, TensorContainer<cpu, 2, double>& input, int beamSize);

private:
    FeatureManager(const FeatureManager &fManager) = delete;
    FeatureManager& operator= (const FeatureManager &fManager) = delete;
};

#endif
