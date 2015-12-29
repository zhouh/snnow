/*************************************************************************
	> File Name: FeatureEmbeddingManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:05:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREEMBEDDINGMANAGER_H_
#define _CHUNKER_FEATUREEMBEDDINGMANAGER_H_

#include <vector>
#include <string>
#include <assert.h>

#include "mshadow/tensor.h"

#include "FeatureEmbedding.h"
#include "FeatureVector.h"
#include "FeatureType.h"
#include "Dictionary.h"
#include "Model.h"
#include "Config.h"

class FeatureEmbeddingManager {
public:
    // std::vector<std::shared_ptr<FeatureEmbedding>> featEmbs;
    std::vector<FeatureType> featTypes;
    std::vector<std::shared_ptr<Dictionary>> featDictPtrs;
    int totalFeatSize;

public:
    FeatureEmbeddingManager(const std::vector<FeatureType> &featTypes, const std::vector<std::shared_ptr<Dictionary>> dictPtrs, const real_t initRange) {
        assert(featTypes.size() == dictPtrs.size());

        totalFeatSize = 0;

        for (auto &fType : featTypes) {
            this->featTypes.push_back(fType);
            totalFeatSize += fType.featSize * fType.featEmbSize;
            // featEmbs.push_back(FeatureEmbedding(fType.featSize, fType.featEmbSize, initRange));
        }

        for (auto &dictPtr : dictPtrs) {
            featDictPtrs.push_back(dictPtr);
        }
    }
    ~FeatureEmbeddingManager() {}

    void readPretrainedEmbeddings(Model<XPU> &model) {
        for (int i = 0; i < static_cast<int>(featTypes.size()); i++) {
            if (featTypes[i].typeName == "word") {
                model.featEmbs[i]->readPreTrain(CConfig::strEmbeddingPath, featDictPtrs[i]->m_mElement2Idx);
            }
        }
    }
    // void readPretrainedEmbeddings(const std::vector<std::pair<bool, std::string>> &isReadPretrainedEmbs) {
    //     for (int i = 0; i < static_cast<int>(isReadPretrainedEmbs.siz()); i++) {
    //         const std::pair<bool, std::string> &isRead = isReadPretrainedEmbs[i];

    //         if (isRead.first) {
    //             featEmbs[i]->readPretrainedEmbeddings(isRead.second, featDictPtrs[i]->m_mElement2Idx);
    //         }
    //     }
    // }

    /**
    * #TODO fill the function
    * convert the input gradients obtained from the neural network
    * to the feature embedding gradients according to the corresponding feature vector
    */
    // void inputGradient2FeatEmbGradient(std::vector<std::shared_ptr<FeatureEmbedding<XPU>>>& featEmbs, FeatureVector& fv, TensorContainer<XPU, 1, real_t>& netInputGradient){

    //     int updateIndex = 0;
    //     for(int j = 0; j < static_cast<int>(fv.features.size()); j++){
    //         
    //         FeatureType ft = featTypes[j];
    //         auto &oneFeatTypeVector = fv.features[j];

    //         for(int i = 0; i < static_cast<int>(oneFeatTypeVector.size()); i++){
    //             featEmbs[j]->data[oneFeatTypeVector[i]] += netInputGradient[ updateIndex ];
    //             updateIndex += ft.featEmbSize;
    //         }
    //     }
    // }

    void returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding<XPU>>> &featEmbs, TensorContainer<XPU, 2, double>& input, int beamSize){
    	// initialize the input
    	input.Resize( Shape2( beamSize, totalFeatSize ), static_cast<real_t>(0.0));
    
    	for(unsigned beamIndex = 0; beamIndex < static_cast<unsigned>(featVecs.size()); beamIndex++) { // for every beam item
            FeatureVector &featVector = featVecs[beamIndex];
    
    		int inputIndex = 0;
            for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featVecs.size()); featTypeIndex++) {
                const std::vector<int> &curFeatVec = featVector[featTypeIndex];
                const int curFeatSize = featTypes[featTypeIndex].featSize;
                const int curEmbSize  = featTypes[featTypeIndex].featEmbSize;
                auto &curFeatEmb = featEmbs[featTypeIndex];
   
                for (int featureIndex = 0; featureIndex < curFeatSize; featureIndex++) {
                    for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                        input[beamIndex][inputIndex++] = curFeatEmb->data[curFeatVec[featureIndex]][embIndex];
                    }
                }
            }
        }
    }

private:
    FeatureEmbeddingManager(const FeatureEmbeddingManager &fEmbManager) = delete;
    FeatureEmbeddingManager& operator= (const FeatureEmbeddingManager &fEmbManager) = delete;
};

#endif
