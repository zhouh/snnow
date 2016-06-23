//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_FEATUREVECTOR_H
#define SNNOW_FEATUREVECTOR_H


#include <vector>
#include <memory>

#include "FeatureType.h"


/**
 *  the feature vector is a 2-ary array, which is used for storing different feature type
 */
class FeatureVector {
public:
    std::vector<std::vector<int>> feature_indexes;

    static FeatureTypes feature_types;

    FeatureVector() {}
    ~FeatureVector() {}

    FeatureVector(const FeatureVector &fv) {
        feature_indexes = fv.feature_indexes;
    }

    FeatureVector&operator=(const FeatureVector &fv) {
        if (this == &fv) {
            return *this;
        }

        feature_indexes = fv.feature_indexes;

        return *this;
    }

    /**
     * static function to set the feature types
     */
    static void setFeatureTypes(FeatureTypes& feature_types) {
        FeatureVector::feature_types = feature_types;
    }

    int size() {
        return feature_indexes.size();
    }

    void clear() {
        feature_indexes.clear();
    }

    void resize(int type_num, int feature_nums[]) {
        feature_indexes.resize(type_num);

        for (int i = 0; i < type_num; ++i) {
            feature_indexes[i].resize(feature_nums[i]);

        }

    }

    //chengc modify
    void resize(std::vector<int> feature_nums) {
        feature_indexes.resize(feature_nums.size());

        for (int i = 0; i < feature_nums.size(); i++) {
            feature_indexes[i].resize(feature_nums[i]);
        }
    }

    void setVector(const int index, std::vector<int> values) {
        feature_indexes[index] = values;
    }

    std::vector<int> getVector(const int index) {
        return feature_indexes[index];
    }

    // const std::vector<int>& operator[] (int index) {
    //     return feature_indexes[index];
    // }

    void push_back(std::vector<int> feature) {
        feature_indexes.push_back(feature);
    }
};


/**
 * The feature vectors is all the feature vectors in an action sequence
 */
typedef  std::vector<FeatureVector> FeatureVectors;


#endif //SNNOW_FEATUREVECTOR_H
