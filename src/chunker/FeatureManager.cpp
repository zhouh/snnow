/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "FeatureManager.h"
#include "Config.h"

void FeatureManager::init(const ChunkedDataSet &goldSet, const std::shared_ptr<DataManager> &dataManagerPtr) {
    int dicSize = 0;
    int featSize = 0;
    int featEmbSize = 0;

    totalFeatSize = 0;

    std::string wordFeatDescription = DataManager::WORDDESCRIPTION;
    std::shared_ptr<DictManager> wordDictManager = dataManagerPtr->m_mStr2DictManager[wordFeatDescription];
    dicSize = wordDictManager->size();
    featSize = 5;
    featEmbSize = 50;
    FeatureType wordFeatType(wordFeatDescription, featSize, featEmbSize);
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new WordFeatureExtractor(
                wordFeatType,
                wordDictManager
                )));
    isReadFeatPretrainedEmbs.push_back(make_pair(true, CConfig::strEmbeddingPath));
    totalFeatSize += featSize * featEmbSize;

    std::string capFeatDescription = DataManager::CAPDESCRIPTION;
    std::shared_ptr<DictManager> capDictManager = dataManagerPtr->m_mStr2DictManager[capFeatDescription];
    dicSize = capDictManager->size();
    featSize = 1;
    featEmbSize = 5;
    FeatureType capFeatType(capFeatDescription, featSize, featEmbSize);
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new CapitalFeatureExtractor(
                capFeatType,
                capDictManager
                )));
    isReadFeatPretrainedEmbs.push_back(make_pair(false, ""));
    totalFeatSize += featSize * featEmbSize;
}

std::vector<FeatureType> FeatureManager::getFeatureTypes() {
    std::vector<FeatureType> featTypes;

    for (auto &fe : featExtractorPtrs) {
        featTypes.push_back(fe->featType);
    }

    return featTypes;
}

std::vector<std::shared_ptr<DictManager>> FeatureManager::getDictManagerPtrs() {
    std::vector<std::shared_ptr<DictManager> dictPtrs;

    for (auto &fe : featExtractorPtrs) {
        dictPtrs.push_back(fe->dictManagerPtr);
    }

    return dictPtrs;
}

void FeatureManager::extractFeature(State &state, Instance &inst, FeatureVector &featVec) {
    for (auto &fe : featExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

void FeatureManager::returnInput(std::vector<FeatureVector> &featVecs, TensorContainer<cpu, 2, double>& input, int beamSize){
	// initialize the input
	input.Resize( Shape2( beamSize, totalFeatSize ), 0.0);

	for(unsigned beamIndex = 0; beamIndex < featVecs.size(); beamIndex++) { // for every beam item
        FeatureVector &featVector = featVecs[beamIndex];

		int inputIndex = 0;
        for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featExtractorPtrs.size()); featTypeIndex++) {
            const std::vector<int> &curFeatVec = featVector[featTypeIndex];
            const int curFeatSize = featExtractorPtrs[featTypeIndex]->featType.featSize;
            const int curEmbSize  = featExtractorPtrs[featTypeIndex]->featType.featEmbSize;
            FeatureEmbedding &curFeatEmb = *(featExtractorPtrs[featTypeIndex]->fEmbPtr.get());

            for (int featureIndex = 0; featureIndex < curFeatSize; featureIndex++) {
                for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                    input[beamIndex][inputIndex++] = curFeatEmb[curFeatVec[featureIndex]][embIndex];
                }
            }
        }
    }
}
