/*************************************************************************
	> File Name: FeatureType.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:06:30 PM CST
 ************************************************************************/

#ifndef _SNNOW_COMMON_FEATURETYPE_H_
#define _SNNOW_COMMON_FEATURETYPE_H_

#include <iostream>
#include <string>
#include <vector>
#include <memory>

class FeatureType;

typedef  std::vector<std::shared_ptr<FeatureType>> FeatureTypes;

/**
 * The Feature Type is used to define the specific feature in a feature embedding framework
 */
class FeatureType {
public:
    std::string type_name;
    int feature_size;
    int dictionary_size;
    int feature_embedding_size;

    const static std::string c_word_type_name = "word";
    const static std::string c_tag_type_name = "tag";
    const static std::string c_label_type_name = "label";
    const static std::string c_capital_type_name = "capital";

public:
    FeatureType(const std::string &name, 
                const int feature_size, 
                const int d_size, 
                const int feat_embed_size) :
            type_name(name),
            feature_size(feature_size),
            dictionary_size(d_size),
            feature_embedding_size(feat_embed_size) {}

    FeatureType(const FeatureType &feat_type) = default;

    FeatureType& operator= (const FeatureType &feat_type) {
        if (this == &feat_type) {
            return *this;
        }

        std::string type_name = feat_type.type_name;
        feature_size = feature_size;
        dictionary_size = dictionary_size;
        feature_embedding_size = feature_embedding_size;

        return *this;
    }

    ~FeatureType() {}
};

inline std::ostream& operator<< (std::ostream &os, const FeatureType &feat_type) {
    os << feat_type.type_name << " " << feat_type.feature_size << " " << feat_type.dictionary_size << " " << feat_type.feature_embedding_size << std::endl;

    return os;
}

inline std::istream& operator>> (std::istream &is, FeatureType &feat_type) {
    is >> feat_type.type_name >> feat_type.feature_size >> feat_type.dictionary_size >> feat_type.feature_embedding_size;

    return is;
}

#endif
