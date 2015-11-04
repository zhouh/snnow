/*
 * Depparser.h
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_DEPPARSER_H_
#define DEPPARSER_DEPPARSER_H_


#define XPU gpu

#include <algorithm>
#include <omp.h>
#include <random>

#include "Config.h"
#include "DepTree.h"
#include "ArcStandardSystem.h"
#include "FeatureExtractor.h"
#include "Beam.h"
#include "BeamDecodor.h"
#include "NNet.h"
#include "TNNets.h"
#include "Evalb.h"


class Depparser {

private:

    FeatureExtractor featExtractor;
    std::vector<GlobalExample> gExamples;
    FeatureEmbedding * fEmb;
    ArcStandardSystem * transitionSystem;

    int beamSize;
    bool m_bTrain;
    bool bEarlyUpdate = true;

public:
	Depparser(bool bTrain);
	Depparser();
	~Depparser();

    // train the input sentences with mini-batch adaGrad
    void train(std::vector<Instance>& instances, std::vector<DepTree>& goldTrees,
            std::vector<Instance>& devInstances, std::vector<DepTree>& devTrees);

    double parse(std::vector<Instance> & devInstances, std::vector<DepTree> & devTree, NNetPara<XPU> & netsPara);

    void trainInit(std::vector<Instance> & trainInstances, std::vector<DepTree> & goldTrees);
};

#endif /* DEPPARSER_DEPPARSER_H_ */
