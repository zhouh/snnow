/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "Config.h"
#include "FeatureManager.h"

void FeatureManager::init(const ChunkedDataSet &goldSet, const std::shared_ptr<DictManager> &dictManagerPtr) {
#define ADDFEATUREEXTRACTOR(name, desc, featSize, featEmbSize, FeatureExtractorType) \
    const std::string name ## Description = desc; \
    const std::shared_ptr<Dictionary> name ## DictPtr = dictManagerPtr->getDictionaryOf(name ## Description); \
    dictSize = name ## DictPtr->size(); \
    FeatureType name ## FeatType(name ## Description, featSize, dictSize, featEmbSize); \
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new FeatureExtractorType(name ## FeatType, name ## DictPtr)))\

    int dictSize = 0;
    int featSize = 0;
    int featEmbSize = 0;

    ADDFEATUREEXTRACTOR(word, DictManager::WORDDESCRIPTION, CConfig::nWordFeatureNum, CConfig::nWordEmbeddingDim, WordFeatureExtractor);

    ADDFEATUREEXTRACTOR(pos, DictManager::POSDESCRIPTION, CConfig::nPOSFeatureNum, CConfig::nPOSEmbeddingDim, POSFeatureExtractor);

    ADDFEATUREEXTRACTOR(cap, DictManager::CAPDESCRIPTION, CConfig::nCapFeatureNum, CConfig::nCapEmbeddingDim, CapitalFeatureExtractor);

#undef ADDFEATUREEXTRACTOR
    // const std::string wordFeatDescription = DictManager::WORDDESCRIPTION;
    // const std::shared_ptr<Dictionary> wordDictPtr = dictManagerPtr->getDictionaryOf(wordFeatDescription);
    // dictSize = wordDictPtr->size();
    // featSize = CConfig::nWordFeatureNum;
    // featEmbSize = CConfig::nWordEmbeddingDim;
    // FeatureType wordFeatType(wordFeatDescription, featSize, dictSize, featEmbSize);
    // featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new WordFeatureExtractor(
    //             wordFeatType,
    //             wordDictPtr
    //             )));

    // const std::string posFeatDescription = DictManager::POSDESCRIPTION;
    // const std::shared_ptr<Dictionary> posDictPtr = dictManagerPtr->getDictionaryOf(posFeatDescription);
    // dictSize = posDictPtr->size();
    // featSize = CConfig::nPOSFeatureNum;
    // featEmbSize = CConfig::nPOSEmbeddingDim;
    // FeatureType posFeatType(posFeatDescription, featSize, dictSize, featEmbSize);
    // featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new POSFeatureExtractor(
    //                 posFeatType,
    //                 posDictPtr
    //                 )));

    // const std::string capFeatDescription = DictManager::CAPDESCRIPTION;
    // const std::shared_ptr<Dictionary> capDictPtr = dictManagerPtr->getDictionaryOf(capFeatDescription);
    // dictSize = capDictPtr->size();
    // featSize = CConfig::nCapFeatureNum;
    // featEmbSize = CConfig::nCapEmbeddingDim;
    // FeatureType capFeatType(capFeatDescription, featSize, dictSize, featEmbSize);
    // featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new CapitalFeatureExtractor(
    //             capFeatType,
    //             capDictPtr
    //             )));
}

std::vector<FeatureType> FeatureManager::getFeatureTypes() {
    std::vector<FeatureType> featTypes;

    for (auto &fe : featExtractorPtrs) {
        featTypes.push_back(fe->getFeatureType());
    }

    return featTypes;
}

std::vector<std::shared_ptr<Dictionary>> FeatureManager::getDictManagerPtrs() {
    std::vector<std::shared_ptr<Dictionary>> dictPtrs;

    for (auto &fe : featExtractorPtrs) {
        dictPtrs.push_back(fe->getDictPtr());
    }

    return dictPtrs;
}

void FeatureManager::extractFeature(const State &state, const Instance &inst, FeatureVector &featVec) {
    for (auto &fe : featExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

