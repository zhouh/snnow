/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "FeatureManager.h"
#include "Config.h"

void FeatureManager::init(const ChunkedDataSet &goldSet, const std::shared_ptr<DictManager> &dictManagerPtr) {
    int dictSize = 0;
    int featSize = 0;
    int featEmbSize = 0;

    totalFeatSize = 0;

    std::string wordFeatDescription = DictManager::WORDDESCRIPTION;
    std::shared_ptr<Dictionary> wordDictPtr = dictManagerPtr->m_mStr2Dict[wordFeatDescription];
    dictSize = wordDictPtr->size();
    featSize = 5;
    featEmbSize = 50;
    FeatureType wordFeatType(wordFeatDescription, featSize, dictSize, featEmbSize);
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new WordFeatureExtractor(
                wordFeatType,
                wordDictPtr
                )));
    totalFeatSize += featSize * featEmbSize;

    std::string capFeatDescription = DictManager::CAPDESCRIPTION;
    std::shared_ptr<Dictionary> capDictPtr = dictManagerPtr->m_mStr2Dict[capFeatDescription];
    dictSize = capDictPtr->size();
    featSize = 1;
    featEmbSize = 5;
    FeatureType capFeatType(capFeatDescription, featSize, dictSize, featEmbSize);
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new CapitalFeatureExtractor(
                capFeatType,
                capDictPtr
                )));
    totalFeatSize += featSize * featEmbSize;
}

std::vector<FeatureType> FeatureManager::getFeatureTypes() {
    std::vector<FeatureType> featTypes;

    for (auto &fe : featExtractorPtrs) {
        featTypes.push_back(fe->featType);
    }

    return featTypes;
}

std::vector<std::shared_ptr<Dictionary>> FeatureManager::getDictManagerPtrs() {
    std::vector<std::shared_ptr<Dictionary>> dictPtrs;

    for (auto &fe : featExtractorPtrs) {
        dictPtrs.push_back(fe->dictPtr);
    }

    return dictPtrs;
}

void FeatureManager::extractFeature(State &state, Instance &inst, FeatureVector &featVec) {
    for (auto &fe : featExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

