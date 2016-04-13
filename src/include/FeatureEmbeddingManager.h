/*************************************************************************
	> File Name: FeatureEmbeddingManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:05:13 PM CST
 ************************************************************************/
#ifndef _SNNOW_FEATUREEMBEDDINGMANAGER_H_
#define _SNNOW_FEATUREEMBEDDINGMANAGER_H_

#include <vector>
#include <string>
#include <memory>
#include <assert.h>

#include "mshadow/tensor.h"
#include "FeatureManager.h"
#include "FeatureEmbedding.h"
#include "FeatureVector.h"
#include "FeatureType.h"
#include "Dictionary.h"
#include "Model.h"

class FeatureEmbeddingManager {
private:
    FeatureTypes feature_types;
    Dicts Dictionary_ptrs;
    int total_feature_embedding_size;

public:
    FeatureEmbeddingManager() { }

    void init(const std::shared_ptr<FeatureManager> &featManagerPtr);

    ~FeatureEmbeddingManager() {}

    std::vector<FeatureType> getFeatureTypes() {
        return m_lFeatTypes;
    }

    int getTotalFeatEmbSize() const {
        return totalFeatEmbSize;
    }

    void readPretrainedEmbeddings(Model<cpu> &model);

    /**
     * According to the feature vectors, extract the input layer for the neural nets
     * from feature embeddings
     *
     * @input the input layer for nets
     */
    void returnInput(FeatureVectors &feature_vectors, FeatureEmbeddings &feature_embeddings, TensorContainer<cpu, 2, real_t>& input);

private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
