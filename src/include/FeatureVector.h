//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_FEATUREVECTOR_H
#define SNNOW_FEATUREVECTOR_H


#include <vector>
#include <memory>

/**
 * The feature vectors is all the feature vectors in an action sequence
 */
typedef  std::vector<std::FeatureVector> FeatureVectors;

/**
 *  the feature vector is a 2-ary array, which is used for storing different feature type
 */
class FeatureVector {
public:
    std::vector<std::vector<int>> feature_indexes;

    FeatureVector() {}
    ~FeatureVector() {}

    FeatureVector(FeatureVector fv) = default;

    int size() {
        return features.size();
    }

    void clear() {
        features.clear();
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


#endif //SNNOW_FEATUREVECTOR_H
