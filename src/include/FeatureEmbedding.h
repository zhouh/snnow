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

using namespace mshadow;

class FeatureEmbedding {
public:
    void saveModel( std::ostream &os ) {
        os << dictSize << "\t" << embeddingSize << std::endl;

        for( int i = 0; i < dictSize; i++ ) {
            for( int j = 0; j < embeddingSize; j++ ){
                os << featEmbeddings[ i ][ j ];
                if( j == ( embeddingSize - 1 ) )
                    os << std::endl;
                else
                    os << " ";
            }
        }
    }

    void loadModel( std::istream & is ){
        std::string line;

        int size0, size1;
        getline( is, line );
        std::istringstream iss(line);
        iss >> size0;
        iss >> size1; 

        for( int i = 0; i < size0; i++  ){
            getline( is, line );
            std::istringstream iss_j( line );
            for( int j = 0; j < size1; j++ )
                iss_j >> featEmbeddings[ i ][ j ];
        }
    }

    FeatureEmbedding(int dictSize, int featureSize, int embeddingSize, double initRange) {
        this->dictSize = dictSize;
        this->featureSize = featureSize;
        this->embeddingSize = embeddingSize;

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-initRange, initRange);

        featEmbeddings.resize(dictSize);

        for (auto &featEmb : featEmbeddings) {
            featEmb.resize(embeddingSize);

            for (int i = 0 ; i < embeddingSize; i++) {
                featEmb[i] = distribution(generator);
            }
        }
    }

    ~FeatureEmbedding() {}

    std::vector<double>& operator[](int featIndex) {
        assert (featIndex >= 0 && featIndex < dictSize);

        return featEmbeddings[featIndex];
    }

    void getPreTrain(int featIndex, std::vector<double> &preTrain) {
        assert (featIndex >= 0 && featIndex < dictSize);
        assert (embeddingSize == preTrain.size());

        for (int i = 0; i < embeddingSize; i++) {
            featEmbeddings[featIndex][i] = preTrain[i];
        }
    }
public:
    std::vector<std::vector<double>> featEmbeddings;
    int dictSize;
    int featureSize;
    int embeddingSize;
};

#endif
