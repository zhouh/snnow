/*************************************************************************
	> File Name: FeatureEmbeddingManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 30 Dec 2015 02:12:04 PM CST
 ************************************************************************/
#include <iostream>
#include <unordered_set>

#include "Config.h"
#include "FeatureEmbeddingManager.h"
#include "FeatureType.h"
#include "mshadow/tensor.h"

void FeatureEmbeddingManager::init(const std::shared_ptr<FeatureManager> &featManagerPtr) {
    std::vector<FeatureType> featTypes = featManagerPtr->getFeatureTypes();
    std::vector<std::shared_ptr<Dictionary>> dictPtrs = featManagerPtr->getDictManagerPtrs();

    totalFeatEmbSize = 0;

    for (auto &fType : featTypes) {
        m_lFeatTypes.push_back(fType);
        totalFeatEmbSize += fType.featSize * fType.featEmbSize;
    }

    for (auto &dictPtr : dictPtrs) {
        m_lFeatDictPtrs.push_back(dictPtr);
    }
}

void FeatureEmbeddingManager::readPretrainedEmbeddings(Model<cpu> &model) {
    for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
        if (m_lFeatTypes[i].typeName == FeatureManager::WORDDESCRIPTION) {
            model.featEmbs[i]->readPreTrain(CConfig::strEmbeddingPath, m_lFeatDictPtrs[i]->getWord2IdxMap());
        }
    }
}

void FeatureEmbeddingManager::returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs, TensorContainer<EMBEDDING_XPU, 2, real_t> &input, int input_offset){
	for(unsigned beamIndex = 0; beamIndex < static_cast<unsigned>(featVecs.size()); beamIndex++) { // for every beam item

        FeatureVector &featVector = featVecs[beamIndex];

		int inputIndex = 0;
        for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featVector.size()); featTypeIndex++) {
            const std::vector<int> &curFeatVec = featVector[featTypeIndex];
            const int curFeatSize = m_lFeatTypes[featTypeIndex].featSize;
            const int curEmbSize  = m_lFeatTypes[featTypeIndex].featEmbSize;
            std::shared_ptr<FeatureEmbedding> &curFeatEmb = featEmbs[featTypeIndex];

            for (auto featId : curFeatVec) {
                Copy(input[input_offset + beamIndex].Slice(inputIndex, inputIndex + curEmbSize), curFeatEmb->data[featId], curFeatEmb->data.stream_);
                inputIndex += curEmbSize;
                // for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                //     input[beamIndex][inputIndex++] = curFeatEmb->data[featId][embIndex];
                // }
            }
        }
    }
}
