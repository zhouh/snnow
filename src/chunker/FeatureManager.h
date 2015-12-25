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
#include <tr1/unordered_map>

#include "FeatureEmbedding.h"
#include "mshadow/tensor.h"

#include "FeatureVector.h"
#include "FeatureType.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"
#include "Example.h"
#include "State.h"

class FeatureManager {
public:
    typedef std::tuple<std::string, int, int, int> FeatureTypeArg;

public:
    FeatureType *labelFeature;
    FeatureType *posFeature;

    std::vector<FeatureType *> featTypes;
    std::vector<FeatureEmbedding *> featEmbs;
    std::vector<int> featSizeOfFeatType;
    int featTypeNum;
    int coreFeatNum;
    int totalFeatSize;
    
    const std::string WORDFEATSTR = "Word Feature";
    int WORDFEATIDX = -1;
    // const std::string POSFEATSTR  = "POS-tag Feature";
    // int POSFEATIDX = -1;
    // const std::string LABELFEATSTR = "Chunk-label Feature";
    // int LABELFEATIDX = -1;
    const std::string CAPFEATSTR = "Word-capital Feature";
    int CAPFEATIDX = -1;

public:
    FeatureManager() { }
    ~FeatureManager() {
        delete labelFeature;
        delete posFeature;
        for (int i = 0; i < featTypes.size(); i++) {
            delete featTypes[i];
        }

        for (int i = 0; i < featEmbs.size(); i++) {
            delete featEmbs[i];
        }
    }

    void init(const ChunkedDataSet &goldSet, double initRange);

    void extractFeature(State &state, Instance &inst, FeatureVector &features);

    void generateInstanceCache(Instance &inst);

    const std::vector<std::string> getKnownLabels() const {
        return labelFeature->getKnownFeatures();
    }

    void generateInstanceSetCache(InstanceSet &instSet) {
        for (auto &inst : instSet) {
            generateInstanceCache(inst);
        }
    }

    int readPretrainEmbeddings(std::string &pretrainFile);

    void generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples);

    /*
     * construct the input by x = beamIndex, y = featureLayerIndex
     */
	void returnInput(std::vector<FeatureVector> &featVecs, TensorContainer<cpu, 2, double>& input, int beamSize);
};

#endif
