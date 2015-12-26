/*
 * FeatureEmbedding.h
 *
 *  Created on: Jul 15, 2015
 *      Author: zhouh
 */

#ifndef INCLUDE_FEATUREEMBEDDING_H_
#define INCLUDE_FEATUREEMBEDDING_H_


#include <sstream>
#include <fstream>
#include "mshadow/tensor.h"


using namespace mshadow;

class FeatureEmbedding {
public:
	FeatureEmbedding(int dicSize, int featureSize, int embeddingSize, int beamSize){

        /*
         * random double [0, 1.0]
         */
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        /*
         * resize the feature embedding
         */
        featEmbeddings.resize(dicSize);
        for(auto & featEmb : featEmbeddings){

            featEmb.resize(embeddingSize);
            for(int i = 0; i < embeddingSize; i++){
                featEmb[i] = distribution(generator);
                //std::cout<<featEmb[i]<<" ";
            }
            //std::cout<<std::endl;
        }

		this->beamSize = beamSize;
		this->inputSize = featureSize * embeddingSize;
        this->embeddingSize = embeddingSize;
	}

	virtual ~FeatureEmbedding(){
	}

    /*
     * get pre-train embedding
     */
    inline void getPreTrain(int embeddingIndex, std::vector<double> & preTrain){
        for(int i = 0; i < preTrain.size(); i++)
            featEmbeddings[ embeddingIndex ][ i ] = preTrain[ i ];
    }

    /*
     * construct the input by x = beamIndex, y = featureLayerIndex
     */
	void returnInput(std::vector< std::vector<int> >& featVecs, TensorContainer<cpu, 2, double>& input){
		// initialize the input
		input.Resize( Shape2( beamSize, inputSize ) );
		for(unsigned beamIndex = 0; beamIndex < featVecs.size(); beamIndex++){ // for every beam item
			int inputIndex = 0;
			for(unsigned featureIndex = 0; featureIndex < featVecs[ beamIndex ].size(); featureIndex++){ // for every feature
                for(unsigned embIndex = 0; embIndex < embeddingSize; embIndex++) // for every doubel in a feature embedding{}
                {
                    if( featVecs[beamIndex][featureIndex] >= featEmbeddings.size() ){

                        std::cout<<"out of mem!"<<featVecs[beamIndex][featureIndex]<<" "<<featEmbeddings.size()<<beamIndex <<" "<<featureIndex<<std::endl;
                    }
                    input[beamIndex][inputIndex++] = featEmbeddings[ featVecs[beamIndex][featureIndex] ][ embIndex ];
                }
			}
		}
	}

    void saveModel( std::ostream & os ){
        os << featEmbeddings.size() << " \t" << featEmbeddings[0].size() << std::endl;
        for( int i = 0; i < featEmbeddings.size(); i++ )
            for( int j = 0; j < featEmbeddings[i].size(); j++ ){
                os << featEmbeddings[ i ][ j ];
                if( j == ( featEmbeddings[i].size() - 1 ) )
                    os << std::endl;
                else
                    os << " ";
            }
    }

    void loadModel( std::istream & is ){

        std::string line;
        int size0, size1;
        getline( is, line );
        std::istringstream iss(line);
        iss >> size0;
        iss >> size1; 
        for( index_t i = 0; i < size0; i++  ){
            getline( is, line );
            std::istringstream iss_j( line );
            for( index_t j = 0; j < size1; j++ )
                iss_j >> featEmbeddings[ i ][ j ];
        }

    }

public:

    std::vector<std::vector<double>> featEmbeddings;
	int beamSize;
	int inputSize;
    int embeddingSize;
};

#endif /* INCLUDE_FEATUREEMBEDDING_H_ */
