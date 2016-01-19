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
    std::vector<std::shared_ptr<FeatureExtractor>> m_lFeatExtractorPtrs;
    std::vector<std::string> m_lEmbeddingNames;
    std::tr1::unordered_map<std::string, std::string> m_mFeatName2EmbeddingName;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;
    static const std::string CHUNKWORDDESCRIPTION;
    static const std::string CHUNKPOSDESCRIPTION;
    
public:
    FeatureManager() { }
    ~FeatureManager() { }

    void init(const ChunkedDataSet &goldSet, const std::shared_ptr<DictManager> &dictaManagerPtr);

    void extractFeature(const State &state, const Instance &inst, FeatureVector &features);

    std::vector<FeatureType> getFeatureTypes();

    std::vector<std::shared_ptr<Dictionary>> getDictManagerPtrs();

    std::string featName2EmbeddingName(const std::string &name);

    std::vector<std::string> getEmebddingNames();

    void saveFeatureManager(std::ostream &os);

    void loadFeatureManager(std::istream &is, const std::shared_ptr<DictManager> &dictManagerPtr);
private:
    FeatureManager(const FeatureManager &fManager) = delete;
    FeatureManager& operator= (const FeatureManager &fManager) = delete;
};

#endif
