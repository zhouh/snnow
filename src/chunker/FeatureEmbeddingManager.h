/*************************************************************************
	> File Name: FeatureEmbeddingManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:05:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREEMBEDDINGMANAGER_H_
#define _CHUNKER_FEATUREEMBEDDINGMANAGER_H_

#include <vector>
#include <assert.h>

#include "FeatureType.h"
#include "DictManger.h"

class FeatureEmbeddingManager {
public:
    std::vector<FeatureEmbedding> featEmbs;
    std::vector<FeatureType> featTypes;
    std::vector<std::shared_ptr<DictManger>> featDictPtrs;

public:
    FeatureEmbeddingManager(const std::vector<FeatureType> &featTypes, const std::vector<std::shared_ptr<DictManager> dictPtrs, const real_t initRange) {
        assert(featTypes.size() == dictPtrs.size());

        for (auto &fType : featTypes) {
            this->featTypes.push_back(fType);
            featEmbs.push_back(FeatureEmbedding(fType.featSize, fType.featEmbSize, initRange));
        }

        for (auto &dictPtr : dictPtrs) {
            featDictPtrs.push_back(dictPtr);
        }
    }
    ~FeatureEmbeddingManager() {}

    void readPretrainedEmbeddings(const std::vector<bool> &bReadEmbs, const std::vector<std::string> &pretrainFiles) {
        for (int i = 0; i < static_cast<int>(bReadEmbs.size()); i++){
            if (bReadEmbs[i]) {
                const std::string &pretrainFile = pretrainFiles[i];

                featEmbs.readPretrainedEmbeddings(pretrainFile, featDictPtrs[i]->m_mElement2Idx);
            }
        }
    }

private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
