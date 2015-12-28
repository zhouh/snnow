/*************************************************************************
	> File Name: FeatureEmbedding_new.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 23 Dec 2015 08:50:11 PM CST
 ************************************************************************/
#ifndef INCLUDE_FEATUREEMBEDDING_H_
#define INCLUDE_FEATUREEMBEDDING_H_

#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <tuple>
#include <assert.h>

#include "mshadow/tensor.h"

#define DEBUG

using namespace mshadow;

template<typename xpu>
class FeatureEmbedding {
public:

    FeatureEmbedding(FeatureTyp& featType) {
        this->dictSize = featType.featSize;
        this->embeddingSize = featType.featEmbSize;

        data.set_stream(stream);
        data.Resize(Shape2(dictSize, embeddingSize), static_cast<real_t>(0.0));  
    }

    void init(real_t initRange){
        Random<xpu, real_t> rnd(0);  
        rnd.SampleUniform(&data, -1.0 * initRange, initRange);
    }

    /**
     * TODO finish this function
     */
    void readPreTrain(std::String sFileName){

    }

    void setZero(){

        data = 0.0;

    }

    ~FeatureEmbedding() {}

    public:
        TensorContainer<xpu, 2, reat_t> data;
        Stream<xpu> *stream = NewStream<xpu>();
        int dictSize;
        int embeddingSize;
    };

#endif
