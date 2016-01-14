/*************************************************************************
	> File Name: FeatureEmbeddingManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:05:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREEMBEDDINGMANAGER_H_
#define _CHUNKER_COMMON_FEATUREEMBEDDINGMANAGER_H_

#include <vector>
#include <string>
#include <memory>
#include <assert.h>

#include "chunker.h"
#include "mshadow/tensor.h"

#include "FeatureManager.h"
#include "FeatureEmbedding.h"
#include "FeatureVector.h"
#include "FeatureType.h"
#include "Dictionary.h"

class FeatureEmbeddingManager {
private:
    std::shared_ptr<FeatureManager> m_featManagerPtr;
    std::vector<FeatureType> m_lFeatTypes;
    std::vector<std::shared_ptr<Dictionary>> m_lFeatDictPtrs;
    std::vector<std::string> m_lEmbeddingNames;
    std::tr1::unordered_map<std::string, int> m_mEmbeddingName2EmbeddingIdx;
    int totalFeatEmbSize;

public:
    FeatureEmbeddingManager(const std::shared_ptr<FeatureManager> &featManagerPtr, const real_t initRange) : m_featManagerPtr(featManagerPtr) {
        std::vector<FeatureType> featTypes = m_featManagerPtr->getFeatureTypes();
        std::vector<std::shared_ptr<Dictionary>> dictPtrs = m_featManagerPtr->getDictManagerPtrs();

        totalFeatEmbSize = 0;

        for (auto &fType : featTypes) {
            m_lFeatTypes.push_back(fType);
            totalFeatEmbSize += fType.featSize * fType.featEmbSize;
        }

        for (auto &dictPtr : dictPtrs) {
            m_lFeatDictPtrs.push_back(dictPtr);
        }

        m_lEmbeddingNames = m_featManagerPtr->getEmebddingNames();
        int idx = 0;
        for (auto &name : m_lEmbeddingNames) {
            m_mEmbeddingName2EmbeddingIdx[name] = idx++;
        }
    }
    ~FeatureEmbeddingManager() {}

    std::vector<FeatureType> getFeatureTypes() {
        return m_lFeatTypes;
    }

    std::vector<std::shared_ptr<FeatureEmbedding>> getInitialzedEmebddings(const real_t initRange) {
        std::vector<std::shared_ptr<FeatureEmbedding>> embeddings(m_lFeatTypes.size());
        std::tr1::unordered_map<std::string, std::shared_ptr<FeatureEmbedding>> name2Embedding;

        for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
            const FeatureType &type = m_lFeatTypes[i];
            const std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

            if (name2Embedding.find(embeddingName) == name2Embedding.end()) {
                name2Embedding[embeddingName].reset(new FeatureEmbedding(type));
            }

            auto found = name2Embedding.find(embeddingName);
            found->second->init(initRange);

            if (m_lFeatTypes[i].typeName == FeatureManager::WORDDESCRIPTION) {
                found->second->readPreTrain(CConfig::strEmbeddingPath, m_lFeatDictPtrs[i]->getWord2IdxMap());
            }
            embeddings[i] = found->second;
        }

        return embeddings;
    }

    std::vector<std::shared_ptr<FeatureEmbedding>> getAllZeroEmebddings() {
        std::vector<std::shared_ptr<FeatureEmbedding>> embeddings(m_lFeatTypes.size());
        std::tr1::unordered_map<std::string, std::shared_ptr<FeatureEmbedding>> name2Embedding;

        for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
            const FeatureType &type = m_lFeatTypes[i];
            const std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

            if (name2Embedding.find(embeddingName) == name2Embedding.end()) {
                name2Embedding[embeddingName] = std::make_shared<FeatureEmbedding>(type);
            }

            auto found = name2Embedding.find(embeddingName);

            embeddings[i] = found->second;
        }

        return embeddings;
    }

    int getTotalFeatEmbSize() const {
        return totalFeatEmbSize;
    }

    void returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs, TensorContainer<cpu, 2, real_t>& input);

private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
