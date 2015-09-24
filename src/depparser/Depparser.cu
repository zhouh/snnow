/*
 * Depparser.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#include <omp.h>
#include <random>
#include <algorithm>

#include "GlobalExample.h"
#include "Depparser.h"
#include "State.h"
#include "Config.h"
#include "NNet.h"
#include "TNNets.h"
#include "FeatureEmbedding.h"
#include "BeamDecodor.h"

using namespace mshadow;
using namespace mshadow::expr;

Depparser::Depparser(bool bTrain) {
    beamSize = CConfig::nBeamSize;
    m_bTrain = bTrain;
}

Depparser::~Depparser() {
}

void Depparser::train(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees,
        std::vector<DepParseInput> devInputs, std::vector<DepTree> devTrees) {

    std::cout<<"Training begin!"<<std::endl;
    std::cout<<"Training Instance Num: "<<inputs.size()<<std::endl;

    /*Prepare the feature extractor*/
    featExtractor.getDictionaries(goldTrees);
    featExtractor.displayDict();
    featExtractor.generateTrainingExamples(inputs, goldTrees, gExamples);

    std::cout << "Constructing dictionary and training examples done!"<<std::endl;

    // prepare for the neural networks, every parsing step maintains a specific net
    // because each parsing step has different updating gradients.
    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = kActNum;
    const int beamSize = CConfig::nBeamSize;
    omp_set_num_threads(CConfig::nThread);  //set the threads for mini-batch learning
    srand(0);
    FeatureEmbedding<cpu> fEmb(CConfig::nFeatureNum, CConfig::nEmbeddingDim, beamSize);
    NNetPara<gpu> netsParas(beamSize, num_in, num_hidden, num_out);

    // for every iteration
    for(int iter = 0; iter < CConfig::nRound; iter++){
        /*
         * randomly sample the training instances in the container,
         * and assign them for each thread
         */
        std::vector<std::vector<GlobalExample*>> multiThread_miniBtach_data;

        //get mini-batch data for each threads
        std::random_shuffle ( gExamples.begin(), gExamples.end() );
        int threadExampleNum = CConfig::nBatchSize / CConfig::nThread;
        auto sp = gExamples.begin();
        auto ep = sp + threadExampleNum;
        for(int i = 0; i < CConfig::nThread; i++){
            std::vector<GlobalExample*> threadExamples;
            for(auto p = sp; p != ep; p++){
                threadExamples.push_back( &( *p ) );
            }
            sp = ep;
            ep += threadExampleNum;
            multiThread_miniBtach_data.push_back(threadExamples);
        }

        std::cout<<"begin to create cuda!";
        
        // begin to multi-thread training
#pragma omp parallel
        {
            auto currentThreadData = multiThread_miniBtach_data[omp_get_thread_num()];
            UpdateGrads<gpu> cumulatedGrads(netsParas.stream);
            TNNets tnnets( beamSize, num_in, num_hidden, num_out, &netParas );

            //for every instance
            for(unsigned inst = 0; inst < currentThreadData.size(); inst++){
                //get current training instance
                GlobalExample * example =  currentThreadData[inst];

                /*
                 * decoding and updating
                 */
                BeamDecodor decodor( example->instance, beamSize, true );
                State * predState = decodor.decoding( tnnets, fEmb, featExtractor, example );
                tnnets.updateTNNetParas( cumulatedGrads, decodor.beam, decodor.bEarlyUpdated, decodor.nGoldTransitIndex, goldScoredTran );
                
            } // instance #for end

            std::cout<<"Begin to update cumulated grads!"<<std::endl;
#pragma omp barrier
#pragma omp critical
            NNet<gpu>::UpdateCumulateGrads(cumulatedGrads, &netsParas);

        } // end multi-processor
    } // iteration #for end
}

void Depparser::parse(std::vector<DepParseInput> inputs) {
}
