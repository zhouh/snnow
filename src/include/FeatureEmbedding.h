#ifndef _SNNOW_FEATUREEMBEDDING_H_
#define _SNNOW_FEATUREEMBEDDING_H_

#include <string>
#include <vector>
#include <tr1/unordered_map>
#include <memory>

#include "Macros.h"
#include "mshadow/tensor.h"
#include "FeatureType.h"

using namespace mshadow;

class FeatureEmbedding;

typedef  std::vector<std::shared_ptr<FeatureEmbedding>> FeatureEmbeddings;

/**
 * the object to store the feature embedding to generate input layer, which needs
 * to be fine-tuned in training
 */
class FeatureEmbedding {

public:
    int dictionary_size;
    int embedding_dim;
    TensorContainer<cpu, 2, real_t> data; // data to store embedding

public:
    FeatureEmbedding(const FeatureType& feature_type) : dictionary_size(feature_type.dictionary_size),
                                                        embedding_dim(feature_type.feature_embedding_size){
        data.Resize(Shape2(dictionary_size, embedding_dim), static_cast<real_t>(0.0));
    }

    ~FeatureEmbedding() {}

    /**
     * init the data of the feature embedding by random
     */
    void init(const real_t init_range);

    void readPreTrain(const std::string &file_name, const std::tr1::unordered_map<std::string, int> &feature_2_idx);

    /**
     * clear all the data
     */
    void setZero(){
        data = 0.0;
    }

private:
    FeatureEmbedding(const FeatureEmbedding &fe) = delete;
    FeatureEmbedding& operator= (const FeatureEmbedding &fe) = delete;
};


#endif
