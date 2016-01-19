/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "Config.h"
#include "FeatureManager.h"

const std::string FeatureManager::WORDDESCRIPTION = "word";
const std::string FeatureManager::POSDESCRIPTION = "pos";
const std::string FeatureManager::LABELDESCRIPTION = "label";
const std::string FeatureManager::CAPDESCRIPTION  = "capital";

void FeatureManager::init(const ChunkedDataSet &goldSet, const std::shared_ptr<DictManager> &dictManagerPtr) {
    int dictSize = 0;
    int featSize = 0;
    int featEmbSize = 0;

#define ADDFEATUREEXTRACTOR(name, dictDesc, featDesc, featSize, featEmbSize, FeatureExtractorType) \
            const std::string name ## Description = dictDesc; \
            const std::shared_ptr<Dictionary> name ## DictPtr = dictManagerPtr->getDictionaryOf(name ## Description); \
            dictSize = name ## DictPtr->size(); \
            FeatureType name ## FeatType(featDesc, featSize, dictSize, featEmbSize); \
            m_lFeatExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new FeatureExtractorType(name ## FeatType, name ## DictPtr))) \
            

    ADDFEATUREEXTRACTOR(word, DictManager::WORDDESCRIPTION, FeatureManager::WORDDESCRIPTION, CConfig::nWordFeatureNum, CConfig::nWordEmbeddingDim, WordFeatureExtractor);

    ADDFEATUREEXTRACTOR(pos, DictManager::POSDESCRIPTION, FeatureManager::POSDESCRIPTION, CConfig::nPOSFeatureNum, CConfig::nPOSEmbeddingDim, POSFeatureExtractor);

    ADDFEATUREEXTRACTOR(cap, DictManager::CAPDESCRIPTION, FeatureManager::CAPDESCRIPTION, CConfig::nCapFeatureNum, CConfig::nCapEmbeddingDim, CapitalFeatureExtractor);

    ADDFEATUREEXTRACTOR(label, DictManager::LABELDESCRIPTION, FeatureManager::LABELDESCRIPTION, CConfig::nLabelFeatureNum, CConfig::nLabelEmbeddingDim, LabelFeatureExtractor);
    // const std::string wordFeatDescription = DictManager::WORDDESCRIPTION;
    // const std::shared_ptr<Dictionary> wordDictPtr = dictManagerPtr->getDictionaryOf(wordFeatDescription);
    // dictSize = wordDictPtr->size();
    // featSize = CConfig::nWordFeatureNum;
    // featEmbSize = CConfig::nWordEmbeddingDim;
    // FeatureType wordFeatType(wordFeatDescription, featSize, dictSize, featEmbSize);
    // m_lFeatExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new WordFeatureExtractor(
    //                 wordFeatType,
    //                 wordDictPtr
    //                 )));

    // const std::string posFeatDescription = DictManager::POSDESCRIPTION;
    // const std::shared_ptr<Dictionary> posDictPtr = dictManagerPtr->getDictionaryOf(posFeatDescription);
    // dictSize = posDictPtr->size();
    // featSize = CConfig::nPOSFeatureNum;
    // featEmbSize = CConfig::nPOSEmbeddingDim;
    // FeatureType posFeatType(posFeatDescription, featSize, dictSize, featEmbSize);
    // m_lFeatExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new POSFeatureExtractor(
    //                 posFeatType,
    //                 posDictPtr
    //                 )));

    // const std::string capFeatDescription = DictManager::CAPDESCRIPTION;
    // const std::shared_ptr<Dictionary> capDictPtr = dictManagerPtr->getDictionaryOf(capFeatDescription);
    // dictSize = capDictPtr->size();
    // featSize = CConfig::nCapFeatureNum;
    // featEmbSize = CConfig::nCapEmbeddingDim;
    // FeatureType capFeatType(capFeatDescription, featSize, dictSize, featEmbSize);
    // m_lFeatExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new CapitalFeatureExtractor(
    //                 capFeatType,
    //                 capDictPtr
    //                 )));

    // const std::string labelFeatDescription = DictManager::LABELDESCRIPTION;
    // const std::shared_ptr<Dictionary> labelDictPtr = dictManagerPtr->getDictionaryOf(labelFeatDescription);
    // dictSize = labelDictPtr->size();
    // featSize = CConfig::nLabelFeatureNum;
    // featEmbSize = CConfig::nLabelEmbeddingDim;
    // FeatureType labelFeatType(labelFeatDescription, featSize, dictSize, featEmbSize);
    // m_lFeatExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new LabelFeatureExtractor(
    //                labelFeatType,
    //                labelDictPtr
    //                )));
}

std::vector<FeatureType> FeatureManager::getFeatureTypes() {
    std::vector<FeatureType> featTypes;

    for (auto &fe : m_lFeatExtractorPtrs) {
        featTypes.push_back(fe->getFeatureType());
    }

    return featTypes;
}

std::vector<std::shared_ptr<Dictionary>> FeatureManager::getDictManagerPtrs() {
    std::vector<std::shared_ptr<Dictionary>> dictPtrs;

    for (auto &fe : m_lFeatExtractorPtrs) {
        dictPtrs.push_back(fe->getDictPtr());
    }

    return dictPtrs;
}

void FeatureManager::extractFeature(const State &state, const Instance &inst, FeatureVector &featVec) {
    for (auto &fe : m_lFeatExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

void FeatureManager::saveFeatureManager(std::ostream &os) {
    os << "featExtractorSize" << " " << m_lFeatExtractorPtrs.size() << std::endl;

    std::vector<FeatureType> featTypes = getFeatureTypes();
    for (FeatureType &ft : featTypes) {
        os << ft;
    }
}

void FeatureManager::loadFeatureManager(std::istream &is, const std::shared_ptr<DictManager> &dictManagerPtr) {
    std::string line;
    std::string tmp;
    getline(is, line);
    std::istringstream iss(line);
    int size;
    iss >> tmp >> size;

    for (int i = 0; i < size; i++) {
        std::string typeName;
        int featSize, dictSize, featEmbSize;

        getline(is, line);
        std::istringstream featIss(line);
        featIss >> typeName >> featSize >> dictSize >> featEmbSize;

        FeatureType ft(typeName, featSize, dictSize, featEmbSize);

        std::shared_ptr<FeatureExtractor> featExtractorPtr;
        if (typeName == FeatureManager::WORDDESCRIPTION) {
            std::string nameDescription = DictManager::WORDDESCRIPTION;
            std::shared_ptr<Dictionary> dictPtr = dictManagerPtr->getDictionaryOf(nameDescription);
            featExtractorPtr.reset(new WordFeatureExtractor(ft, dictPtr));
        } else if (typeName == FeatureManager::POSDESCRIPTION) {
            std::string nameDescription = DictManager::POSDESCRIPTION;
            std::shared_ptr<Dictionary> dictPtr = dictManagerPtr->getDictionaryOf(nameDescription);
            featExtractorPtr.reset(new POSFeatureExtractor(ft, dictPtr));
        } else if (typeName == FeatureManager::CAPDESCRIPTION) {
            std::string nameDescription = DictManager::CAPDESCRIPTION;
            std::shared_ptr<Dictionary> dictPtr = dictManagerPtr->getDictionaryOf(nameDescription);
            featExtractorPtr.reset(new CapitalFeatureExtractor(ft, dictPtr));
        } else if (typeName == FeatureManager::LABELDESCRIPTION) {
            std::string nameDescription = DictManager::LABELDESCRIPTION;
            std::shared_ptr<Dictionary> dictPtr = dictManagerPtr->getDictionaryOf(nameDescription);
            featExtractorPtr.reset(new LabelFeatureExtractor(ft, dictPtr));
        } else {
            std::cerr << "load wrong FeatureType: " << typeName << std::endl;
            exit(0);
        }

        m_lFeatExtractorPtrs.push_back(featExtractorPtr);
    }
}
