/*
 * Depparser.h
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_DEPPARSER_H_
#define DEPPARSER_DEPPARSER_H_

#include <algorithm>

#include "DepTree.h"
#include "State.h"
#include "GlobalExample.h"
#include "Example.h"
#include "FeatureExtractor.h"
#include "Beam.h"

class Depparser {

private:

    FeatureExtractor featExtractor;
    std::vector<GlobalExample> gExamples;

    int beamSize;
    bool m_bTrain;
    bool bEarlyUpdate = true;

public:
	Depparser(bool bTrain);
	Depparser();
	~Depparser();

    void init(std::vector<DepParseInput> inputs,
            std::vector<DepTree> goldTrees);
	
    // train the input sentences with mini-batch adaGrad
    void train(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees,
            std::vector<DepParseInput> devInputs, std::vector<DepTree> devTrees);

    void parse(std::vector<DepParseInput> inputs);

private:
    void generateTrainingExamples(std::vector<DepParseInput> inputs,
            std::vector<DepTree> goldTrees);

    void getInputBatch(State* state, std::vector<int>& wordIndexCache,
            std::vector<int>& tagIndexCache, std::vector< std::vector<int> >& features);
};

#endif /* DEPPARSER_DEPPARSER_H_ */
