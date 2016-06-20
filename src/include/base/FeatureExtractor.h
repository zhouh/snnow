/*************************************************************************
	> File Name: Feature.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 03:48:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREEXTRACTOR_H_
#define _CHUNKER_COMMON_FEATUREEXTRACTOR_H_

#include <string>
#include <memory>

#include "State.h"
#include "Input.h"
#include "FeatureType.h"
#include "FeatureVector.h"

class FeatureExtractor{

public:
    FeatureExtractor() = default;

    virtual ~FeatureExtractor() {}

    virtual FeatureVector getFeatureVectors(State& state, Input& input ) = 0;

    virtual void getDictionaries(DataSet& data) = 0;

private:
    FeatureExtractor(const FeatureExtractor &fe) = delete;
    FeatureExtractor& operator= (const FeatureExtractor &fe) = delete;
};


#endif
