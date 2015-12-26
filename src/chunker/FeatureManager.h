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
#include <memory>
#include <tr1/unordered_map>

#include "mshadow/tensor.h"

#include "FeatureExtractor.h"
#include "ChunkerDataManager.h"

#include "FeatureVector.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"
#include "Example.h"
#include "State.h"

class FeatureManager {
public:
    ChunkerDataManager dataManager;
    std::vector<std::shared_ptr<FeatureExtractor>> featExtractorPtrs;
    int totalFeatSize;

public:
    FeatureManager() { }
    ~FeatureManager() { }

    void init(const ChunkedDataSet &goldSet, double initRange, const bool readPretrainEmbs = false, const std::string &pretrainFile = "");

    void extractFeature(State &state, Instance &inst, FeatureVector &features);

    void generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples);

    void generateInstanceSetCache(InstanceSet &instSet) {
        for (auto &inst : instSet) {
            dataManager.generateInstanceCache(inst);
        }
    }

    /*
     * construct the input by x = beamIndex, y = featureLayerIndex
     */
	void returnInput(std::vector<FeatureVector> &featVecs, TensorContainer<cpu, 2, double>& input, int beamSize);

private:
    int readPretrainedEmbeddings(const std::string &pretrainFile, const std::tr1::unordered_map<std::string, int> &word2IdxMap, FeatureEmbedding *fEmb);
};

#endif
