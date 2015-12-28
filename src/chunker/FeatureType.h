/*************************************************************************
	> File Name: FeatureType.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 28 Dec 2015 01:06:30 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATURETYPE_H_
#define _CHUNKER_FEATURETYPE_H_

#include <string>

class FeatureType {
public:
    const std::string typeName;
    const int featSize;
    const int featEmbSize;

public:
    FeatureType(const std::string &name, const int featureSize, const int featureEmbSize) : typeName(name), featSize(featureSize), featEmbSize(featureEmbSize) {}

    FeatureType(const FeatureType &fType) : typeName(fType.typeName), featSize(fType.featSize), featEmbSize(fType.featEmbSize) {}

    FeatureType& operator= (const FeatureType &fType) {
        if (this == &fType) {
            return *this;
        }

        typeName = fType.typeName;
        featSize = fType.featSize;
        featEmbSize = fType.featEmbSize;

        return *this;
    }

    ~FeatureType() {}
};

#endif
