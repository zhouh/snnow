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

typedef double real_t;

#include "FeatureType.h"
#include "mshadow/tensor.h"

#define DEBUG

using namespace mshadow;

template<typename xpu>
class FeatureEmbedding {
public:

    FeatureEmbedding(FeatureType& featType) {
        this->dictSize = featType.dictSize;
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
    void readPreTrain(std::string sFileName, std::tr1::unordered_map<std::string, int> &word2Idx){
        std::tr1::unordered_map<std::string, int> pretrainWords;
        std::vector<std::vector<real_t>> pretrainEmbs;
        std::string line;
        std::ifstream in(sFileName);

        int index = 0;
        while (getline(in, line)) {
            if (line.empty()) {
                continue;
            }

            std::istringstream iss(line);
            std::vector<real_t> embedding;

            std::string word;
            real_t d;
            iss >> word;
            while (iss >> d) {
                embedding.push_back(d);
            }

            pretrainEmbs.push_back(embedding);
            pretrainWords[word] = index++;
        }

        std::cerr << "  pretrainWords's size: " << pretrainEmbs.size() << std::endl;

        for (auto &wordPair : word2Idx) {
            auto ret = pretrainWords.find(wordPair.first);

            if (pretrainWords.end() != ret) {
                int featIndex  = wordPair.second;
                auto &preTrain = pretrainEmbs[ret->second];

                assert (featIndex >= 0 && featIndex < dictSize);
                assert (embeddingSize == static_cast<int>(preTrain.size()));

                for (int i = 0; i < embeddingSize; i++) {
                    data[featIndex][i] = preTrain[i];
                }
            }
        }
    }

    void setZero(){
        data = 0.0;
    }

    ~FeatureEmbedding() {}

public:
    TensorContainer<xpu, 2, real_t> data;
    Stream<xpu> *stream = NewStream<xpu>();
    int dictSize;
    int embeddingSize;
};

#endif
