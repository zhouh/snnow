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

#include "FeatureEmbedding.h"
#include "FeatureVector.h"
#include "FeatureType.h"
#include "Dictionary.h"
#include "Model.h"

class FeatureEmbeddingManager {
private:
    std::vector<FeatureType> m_lFeatTypes;
    std::vector<std::shared_ptr<Dictionary>> m_lFeatDictPtrs;
    int totalFeatEmbSize;

public:
    FeatureEmbeddingManager(const std::vector<FeatureType> &featTypes, const std::vector<std::shared_ptr<Dictionary>> dictPtrs, const real_t initRange) {
        assert(featTypes.size() == dictPtrs.size());

        totalFeatEmbSize = 0;

        for (auto &fType : featTypes) {
            m_lFeatTypes.push_back(fType);
            totalFeatEmbSize += fType.featSize * fType.featEmbSize;
        }

        for (auto &dictPtr : dictPtrs) {
            m_lFeatDictPtrs.push_back(dictPtr);
        }
    }
    ~FeatureEmbeddingManager() {}

    int getTotalFeatEmbSize() const {
        return totalFeatEmbSize;
    }

    void readPretrainedEmbeddings(Model<XPU> &model);

    void returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs, TensorContainer<cpu, 2, real_t>& input);

private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
