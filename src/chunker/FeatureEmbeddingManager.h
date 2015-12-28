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
    std::vector<std::shared_ptr<FeatureEmbedding>> featEmbs;
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

    /**
     * #TODO fill the function
     * convert the input gradients obtained from the neural network
     * to the feature embedding gradients according to the corresponding feature vector
     */
    void inputGradient2FeatEmbGradient(std::vector<std::shared_ptr<FeatureEmbedding>>& featEmbs, FeatureVector& fv, TensorContainer<1, real_t>& netInputGradient){

        int updateIndex = 0;
        for(int j = 0; j < fv.features.size(); i++){
            
            FeatureType ft = featureTypes[j];
            auto oneFeatTypeVector = fv.features[j];

            for(int i = 0; i < oneFeatTypeVector.size(); i++){
                featEmbs[j]->data[oneFeatTypeVector[i]][dim] += netInputGradient[ updateIndex ];
                updateIndex += ft.featEmbSize;
            }
        }
    }


private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
