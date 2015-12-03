/*
 * Depparser.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */
#include <ctime>
#include <memory>
#include "Depparser.h"

using namespace mshadow;
using namespace mshadow::expr;

Depparser::Depparser(bool bTrain) {
    beamSize = CConfig::nBeamSize;
    m_bTrain = bTrain;
    transitionSystem = new ArcStandardSystem();
}

Depparser::~Depparser() {
    delete fEmb;
    delete transitionSystem;
}

void Depparser::trainInit(std::vector<Instance> & trainInstances, std::vector<DepTree> & goldTrees){

    std::cout<<"Training begin!"<<std::endl;
    std::cout<<"Training Instance Num: "<<trainInstances.size()<<std::endl;

    /*Prepare the feature extractor*/
    featExtractor.getDictionaries(goldTrees);
    transitionSystem->makeTransition(featExtractor.knowLabels);
    featExtractor.displayDict();
    fEmb = new FeatureEmbedding(featExtractor.getDicSize(), CConfig::nFeatureNum, CConfig::nEmbeddingDim, beamSize);
    featExtractor.generateTrainingExamples(transitionSystem, trainInstances, goldTrees, gExamples);
    /*
     * prepare the feature embedding and fill in pre-train embedding
     */
    featExtractor.readPretrainEmbeddings( CConfig::strEmbeddingPath, *fEmb );

    std::cout << "Constructing dictionary and training examples done!"<<std::endl;
}

void Depparser::train(std::vector<Instance> & trainInstances, std::vector<DepTree> & goldTrees,
        std::vector<Instance> & devInstances, std::vector<DepTree> & devTrees) {
    
    trainInit(trainInstances, goldTrees);
    featExtractor.getInstancesCache(devInstances); // train instances get cache in get training examples
    /*
     * prepare for the neural networks, every parsing step maintains a specific net
     * because each parsing step has different updating gradients.
     */
    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = transitionSystem->nActNum;
    const int beamSize = CConfig::nBeamSize;
    omp_set_num_threads(CConfig::nThread);  //set the threads for mini-batch learning
    srand(0);
    // std::shared_ptr< NNetPara<XPU> > ptrNetsParas(new NNetPara<XPU>(beamSize, num_in, num_hidden, num_out));
    NNetPara<XPU> netsParas(beamSize, num_in, num_hidden, num_out);
    double bestdevUAS = -1.0;

    // for every iteration
    for(int iter = 0; iter < CConfig::nRound; iter++){

        /*
         * Evaluate per iterations
         */
        if( (iter % CConfig::nEvaluatePerIters) == 0 ){
            double currentUAS = parse( devInstances, devTrees, netsParas );
            if( currentUAS > bestdevUAS )
                bestdevUAS = currentUAS;
            std::cout<<"current iteration UAS: "<<currentUAS<<" new best UAS:\t"<< bestdevUAS<<std::endl;;
        }

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
            for(auto p = sp; p != ep; p++)
                threadExamples.push_back( &( *p ) );
            sp = ep;
            ep += threadExampleNum;
            multiThread_miniBtach_data.push_back(threadExamples);
        }

        /*std::cout<<"begin to create cuda!"<<std::endl;*/
        
        // begin to multi-thread training
#pragma omp parallel
        {
            /*cudaSetDevice( omp_get_thread_num() % 4 );*/
            auto currentThreadData = multiThread_miniBtach_data[omp_get_thread_num()];
            UpdateGrads<XPU> cumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);

            //for every instance
            for(unsigned inst = 0; inst < currentThreadData.size(); inst++){
                //get current training instance
                GlobalExample * example =  currentThreadData[inst];

                TensorContainer<gpu,2, real_t> mask;
                mask.set_stream(netsParas.stream);
                netsParas.rnd.SampleUniform(&mask, 0.0f, 1.0f);

                TNNets tnnets( beamSize, num_in, num_hidden, num_out, &netsParas);
                /*
                 * decoding and updating
                 */
                std::cout<<"begin to decod!"<<std::endl;
                BeamDecodor decodor( &( example->instance ), beamSize, true );
                
                std::cout<<"end to decod!"<<std::endl;
                State * predState = decodor.decoding( transitionSystem, tnnets, featExtractor, *fEmb, example );
                tnnets.updateTNNetParas( cumulatedGrads, decodor.beam, decodor.bEarlyUpdated, decodor.nGoldTransitIndex, decodor.goldScoredTran );
                
                /*std::cout<<"begin to back subsidegrads!"<<std::endl;*/
            } // instance #for end

            /*std::cout<<"Begin to update cumulated grads!"<<std::endl;*/
#pragma omp barrier
#pragma omp critical
            NNet<XPU>::UpdateCumulateGrads(cumulatedGrads, &netsParas);

        } // end multi-processor
    } // iteration #for end
}

double Depparser::parse( std::vector<Instance> & devInstances, std::vector<DepTree> & devTree, NNetPara<XPU> & netsParas){

            const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
            const int num_hidden = CConfig::nHiddenSize;
            const int num_out = transitionSystem->nActNum;
            const int beamSize = CConfig::nBeamSize;
            TNNets tnnets( beamSize, num_in, num_hidden, num_out, &netsParas, false );

            std::vector<DepTree> predTrees(devInstances.size());

            clock_t start, end; 
            start = clock();
            //for every instance
            for(unsigned inst = 0; inst < devInstances.size(); inst++){
                predTrees[inst].init(devInstances[inst].input);
                /*std::cout<<"instance\t"<<inst<<std::endl;*/
                //get current training instance
                BeamDecodor decodor( &( devInstances[inst] ), beamSize, false );
                State * predState = decodor.decoding( transitionSystem, tnnets, featExtractor, *fEmb);
                transitionSystem->GenerateOutput( *predState, devInstances[inst].input, predTrees[inst] );
            } // instance #for end
            end = clock(); 
            double timeuse = (double)(end - start) / CLOCKS_PER_SEC;
            std::cout<<"totally parse "<<devInstances.size()<<" sentences, time : "<< timeuse << " average: "<< devInstances.size()/timeuse<<" sentences/second!"<<std::endl;
            /*
             * evaluate
             */
            auto result = Evalb::evalb(predTrees, devTree);

            return result.first * 100;
}
