/*************************************************************************
	> File Name: FeatureVector.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 07:47:36 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREVECTOR_H_
#define _CHUNKER_COMMON_FEATUREVECTOR_H_

#include <vector>

class FeatureVector {
public:
    std::vector<std::vector<int>> features;

    FeatureVector() {}
    ~FeatureVector() {}

    int size() {
        return features.size();
    }

    void resize(int capacity) {
        features.resize(capacity);
    }

    std::vector<int>& operator[] (int index) {
        return features[index];
    }

    void push_back(std::vector<int> feature) {
        features.push_back(feature);
    }
};

#endif
