/*************************************************************************
	> File Name: FeatureEmbeddingManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 30 Dec 2015 02:12:04 PM CST
 ************************************************************************/
#include <iostream>

#include "Config.h"
#include "FeatureEmbeddingManager.h"

void FeatureEmbeddingManager::returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs, TensorContainer<cpu, 2, real_t> &input){
    // TODO: if neccessary ?
	// initialize the input
	// input.Resize( Shape2( beamSize, totalFeatEmbSize ), static_cast<real_t>(0.0));

	for(unsigned beamIndex = 0; beamIndex < static_cast<unsigned>(featVecs.size()); beamIndex++) { // for every beam item

        FeatureVector &featVector = featVecs[beamIndex];

		int inputIndex = 0;
        for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featVector.size()); featTypeIndex++) {
            const std::vector<int> &curFeatVec = featVector[featTypeIndex];
            const int curFeatSize = m_lFeatTypes[featTypeIndex].featSize;
            const int curEmbSize  = m_lFeatTypes[featTypeIndex].featEmbSize;
            std::shared_ptr<FeatureEmbedding> &curFeatEmb = featEmbs[featTypeIndex];

            for (auto featId : curFeatVec) {
                for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                    // std::cerr << "[data]dim0 size: " << curFeatEmb->data.shape_[0] << std::endl;
                    // std::cerr << "featId: " << featId << std::endl;
                    // std::cerr << "[data]dim1 size: " << curFeatEmb->data.shape_[1] << std::endl;
                    // std::cerr << "embIndex: " << embIndex << std::endl;
                    // std::cerr << "[input]dim0 size: " << input.shape_[0] << std::endl;
                    // std::cerr << "beamIndex: " << beamIndex << std::endl;
                    // std::cerr << "[input]dim1 size: " << input.shape_[1] << std::endl;
                    // std::cerr << "inputIndex: " << inputIndex << std::endl;
                    input[beamIndex][inputIndex++] = curFeatEmb->data[featId][embIndex];
                    // curFeatEmb->data[featId][embIndex];
                }
            }
            // for (int featureIndex = 0; featureIndex < curFeatSize; featureIndex++) {
            //     for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
            //         input[beamIndex][inputIndex++] = curFeatEmb->data[curFeatVec[featureIndex]][embIndex];
            //     }
            // }
        }
    }
}
