/*************************************************************************
	> File Name: FeatureVector.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 07:47:36 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREVECTOR_H_
#define _CHUNKER_FEATUREVECTOR_H_

#include <vector>

#include "FeatureType.h"
#include "FeatureEmbedding.h"

class FeatureVector {
public:
    const std::vector<FeatureType *> &featTypes;
    const std::vector<FeatureEmbedding *> &featEmbs;
    std::vector<std::vector<int>> features;

    FeatureVector(const std::vector<FeatureType *> &featureTypes, const std::vector<FeatureEmbedding *> &featureEmbs) : featTypes(featureTypes), featEmbs(featureEmbs) { 
    
    }
    FeatureVector(const FeatureVector &featVec) : featTypes(featVec.featTypes), featEmbs(featVec.featEmbs), features(featVec.features) {

    }
    FeatureVector& operator= (const FeatureVector &featVec) = delete;

    ~FeatureVector() {}

    int size() {
        return static_cast<int>(featTypes.size());
    }

    void resize(int capacity) {
        features.resize(capacity);
    }

    std::vector<int>& operator[] (int index) {
        return features[index];
    }

    void push_back(std::vector<int> &feature) {
        features.push_back(feature);
    }
};

#endif
