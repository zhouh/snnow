/*************************************************************************
	> File Name: FeatureType.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:06:30 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATURETYPE_H_
#define _CHUNKER_COMMON_FEATURETYPE_H_

#include <iostream>
#include <string>

class FeatureType {
public:
    std::string typeName;
    int featSize;
    int dictSize;
    int featEmbSize;

public:
    FeatureType(const std::string &name, const int featureSize, const int dictionarySize, const int featureEmbSize) : typeName(name), featSize(featureSize), dictSize(dictionarySize), featEmbSize(featureEmbSize) {}

    FeatureType(const FeatureType &fType) : typeName(fType.typeName), featSize(fType.featSize), dictSize(fType.dictSize), featEmbSize(fType.featEmbSize) {}

    FeatureType& operator= (const FeatureType &fType) {
        if (this == &fType) {
            return *this;
        }

        typeName = fType.typeName;
        featSize = fType.featSize;
        dictSize = fType.dictSize;
        featEmbSize = fType.featEmbSize;

        return *this;
    }

    ~FeatureType() {}
};

inline std::ostream& operator<< (std::ostream &os, const FeatureType &fType) {
    os << fType.typeName << " " << fType.featSize << " " << fType.dictSize << " " << fType.featEmbSize << std::endl;

    return os;
}

inline std::istream& operator>> (std::istream &is, FeatureType &fType) {
    is >> fType.typeName >> fType.featSize >> fType.dictSize >> fType.featEmbSize;

    return is;
}
#endif
