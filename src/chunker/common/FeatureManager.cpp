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
const std::string FeatureManager::CHUNKWORDDESCRIPTION = "chunkword";
const std::string FeatureManager::CHUNKPOSDESCRIPTION = "chunkpos";

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

    ADDFEATUREEXTRACTOR(cap, DictManager::CAPDESCRIPTION, FeatureManager::LABELDESCRIPTION, CConfig::nCapFeatureNum, CConfig::nCapEmbeddingDim, CapitalFeatureExtractor);

    ADDFEATUREEXTRACTOR(label, DictManager::LABELDESCRIPTION, FeatureManager::CAPDESCRIPTION, CConfig::nLabelFeatureNum, CConfig::nLabelEmbeddingDim, LabelFeatureExtractor);

    ADDFEATUREEXTRACTOR(chunkword, DictManager::WORDDESCRIPTION, FeatureManager::CHUNKWORDDESCRIPTION, CConfig::nChunkWordFeatureNum, CConfig::nChunkWordEmbeddingDim, ChunkWordFeatureExtractor);

    ADDFEATUREEXTRACTOR(chunkpos, DictManager::POSDESCRIPTION, FeatureManager::CHUNKPOSDESCRIPTION, CConfig::nChunkPOSFeatureNum, CConfig::nChunkPOSEmbeddingDim, ChunkPOSFeatureExtractor);
#undef ADDFEATUREEXTRACTOR

    m_lEmbeddingNames.push_back(WORDDESCRIPTION);
    m_lEmbeddingNames.push_back(POSDESCRIPTION);
    m_lEmbeddingNames.push_back(LABELDESCRIPTION);
    m_lEmbeddingNames.push_back(CAPDESCRIPTION);
    m_mFeatName2EmbeddingName[WORDDESCRIPTION]      = WORDDESCRIPTION;
    m_mFeatName2EmbeddingName[POSDESCRIPTION]       = POSDESCRIPTION;
    m_mFeatName2EmbeddingName[LABELDESCRIPTION]     = LABELDESCRIPTION;
    m_mFeatName2EmbeddingName[CAPDESCRIPTION]       = CAPDESCRIPTION;
    m_mFeatName2EmbeddingName[CHUNKWORDDESCRIPTION] = WORDDESCRIPTION;
    m_mFeatName2EmbeddingName[CHUNKPOSDESCRIPTION]  = POSDESCRIPTION;
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

std::string FeatureManager::featName2EmbeddingName(const std::string &name) {
    auto found = m_mFeatName2EmbeddingName.find(name);
    if (found == m_mFeatName2EmbeddingName.end()) {
        std::cerr << name << " can not be mapped to embeddingname!" << std::endl;

        exit(0);
    }

    return found->second;
}

std::vector<std::string> FeatureManager::getEmebddingNames() {
    std::vector<std::string> ret(m_lEmbeddingNames);

    return ret;
}

void FeatureManager::extractFeature(const State &state, const Instance &inst, FeatureVector &featVec) {
    for (auto &fe : m_lFeatExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

