/*************************************************************************
	> File Name: FeatureEmbedding.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 23 Dec 2015 08:50:11 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREEMBEDDING_H_
#define _CHUNKER_COMMON_FEATUREEMBEDDING_H_

#include <string>
#include <tr1/unordered_map>

#include "chunker.h"

#include "mshadow/tensor.h"

#include "FeatureType.h"

using namespace mshadow;

class FeatureEmbedding {
public:
    FeatureEmbedding(const FeatureType& featType, Stream<EMBEDDING_XPU> *stream) : dictSize(featType.dictSize), embeddingSize(featType.featEmbSize){
        data.set_stream(stream);
        data.Resize(Shape2(dictSize, embeddingSize), static_cast<real_t>(0.0));
    }

    ~FeatureEmbedding() {}

    void init(const real_t initRange);

    void readPreTrain(const std::string &sFileName, const std::tr1::unordered_map<std::string, int> &word2Idx);

    void setZero(){
        data = 0.0;
    }

public:
    int dictSize;
    int embeddingSize;
    TensorContainer<EMBEDDING_XPU, 2, real_t> data;

private:
    FeatureEmbedding(const FeatureEmbedding &fe) = delete;
    FeatureEmbedding& operator= (const FeatureEmbedding &fe) = delete;
};

#endif
