/*
 * FeatureEmbedding.h
 *
 *  Created on: Jul 15, 2015
 *      Author: zhouh
 */

#ifndef INCLUDE_FEATUREEMBEDDING_H_
#define INCLUDE_FEATUREEMBEDDING_H_

#include "mshadow/tensor.h"

using namespace mshadow;

template<typename xpu>
class FeatureEmbedding {
public:
	FeatureEmbedding(int featureNum, int embeddingSize, int beamSize) : rnd(0){
		featEmbeddings.Resize( Shape2( featureNum, embeddingSize ) );
		rnd.SampleGaussian( &featEmbeddings, 0, CConfig::fInitRange );

		this->beamSize = beamSize;
		this->inputSize = featureNum * embeddingSize;
	}
	virtual ~FeatureEmbedding(){
	}

	void returnInput(std::vector< std::vector<int> >& featVecs, TensorContainer<xpu, 2>& input){
		// initialize the input
		input.Resize( Shape2( beamSize, inputSize ) );
		for(unsigned i = 0; i < featVecs.size(); i++){
			int inputIndex = 0;
			for(unsigned j = 0; j < featVecs[i].size(); j++){
				input[i][inputIndex++] = featEmbeddings[ featVecs[i][j] ][j];
			}
		}
	}

public:
	TensorContainer<xpu,2> featEmbeddings;
	Random<xpu> rnd;
	int beamSize;
	int inputSize;
};

#endif /* INCLUDE_FEATUREEMBEDDING_H_ */
