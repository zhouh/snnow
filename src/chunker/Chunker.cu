/*************************************************************************
	> File Name: Chunker.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 03:59:17 PM CST
 ************************************************************************/
#include <ctime>
#include <omp.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits.h>

#include "Config.h"

#include "Chunker.h"
#include "TNNets.h"
#include "Evalb.h"

#define DEBUG

#ifdef DEBUG
// #define DEBUG1
// #define DEBUG2
// #define DEBUG4
// #define DEBUG5
#define CHECKNETVALUES
#endif

Chunker::Chunker() {
    m_nBeamSize = CConfig::nBeamSize;
    m_bTrain = false;
    m_bEarlyUpdate = false;
}

Chunker::Chunker(bool isTrain) {
    m_nBeamSize = CConfig::nBeamSize;
    m_bTrain = isTrain;
    m_bEarlyUpdate = false;
}

Chunker::~Chunker() {

}
    
double Chunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsParas) {
    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transitionSystem->nActNum;
    const int beam_size = CConfig::nBeamSize;

    TNNets tnnets(beam_size, num_in, num_hidden, num_out, &netsParas, false);

    clock_t start, end;
    start = clock();
    ChunkedDataSet predictDevSet;
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        ChunkedSentence predictSent(devInstances[inst].input);

        BeamDecoder decoder(&(devInstances[inst]), 
                            m_transitionSystem.get(),
                            m_featExtractor.get(),
                            m_fEmb.get(),
                            m_nBeamSize, 
                            false);

        decoder.generateChunkedSentence(tnnets, predictSent);

        predictDevSet.push_back(predictSent);
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr << "totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

    auto res = Evalb::eval(predictDevSet, goldDevSet);

    return std::get<2>(res);
}

void Chunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    initTrain(trainGoldSet, trainSet);

    m_featExtractor->generateInstanceSetCache(devSet);

    m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);

    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transitionSystem->nActNum;
    const int batch_size = std::min(CConfig::nBatchSize, static_cast<int>(gExamples.size()));

    const int beam_size = CConfig::nBeamSize;

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    NNetPara<XPU> netsParas(beam_size, num_in, num_hidden, num_out);

    double bestDevFB1 = -1.0;
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double currentFB1 = chunk(devSet, devGoldSet, netsParas);
            if (currentFB1 > bestDevFB1) {
                bestDevFB1 = currentFB1;
            }
            auto sf = std::cerr.flags();
            auto sp = std::cerr.precision();
            std::cerr.flags(std::ios::fixed);
            std::cerr.precision(2);
            std::cerr << "current iteration FB1-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << currentFB1 << "\t best FB1-score: " << bestDevFB1 << std::endl;
            std::cerr.flags(sf);
            std::cerr.precision(sp);
        }

        // random shuffle the training instances in the container,
        // and assign them for each thread
        std::vector<std::vector<GlobalExample *>> multiThread_miniBatch_data;

        // prepare mini-batch data for each threads
        std::random_shuffle(gExamples.begin(), gExamples.end());
        int exampleNumOfThread = batch_size / CConfig::nThread;
        auto sp = gExamples.begin();
        auto ep = sp + exampleNumOfThread;
        for (int i = 0; i < CConfig::nThread; i++) {
            std::vector<GlobalExample *> threadExamples;
            for (auto p = sp; p != ep; p++) {
                threadExamples.push_back(&(*p));
            }
            sp = ep;
            ep += exampleNumOfThread;
            multiThread_miniBatch_data.push_back(threadExamples);
        }

        UpdateGrads<XPU> batchCumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);
        // begin to multi thread Training
#pragma omp parallel
        {
            auto currentThreadData = multiThread_miniBatch_data[omp_get_thread_num()];
            UpdateGrads<XPU> threadCumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);

            // for evary instance in this mini-batch
            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                // fetch a to-be-trained instance
                GlobalExample *example = currentThreadData[inst];

                TNNets tnnets(m_nBeamSize, num_in, num_hidden, num_out, &netsParas);

                // decode and update
                // std::cerr << "begin to decode!" << std::endl;
                BeamDecoder decoder(&(example->instance), 
                                    m_transitionSystem.get(),
                                    m_featExtractor.get(),
                                    m_fEmb.get(),
                                    m_nBeamSize, 
                                    true);

                State * predState = decoder.decode(tnnets, example);

                tnnets.updateTNNetParas(threadCumulatedGrads, decoder.beam, decoder.bEarlyUpdate, decoder.nGoldTransitionIndex, decoder.goldScoredTran);
            } // end for instance traverse

#pragma omp barrier
#pragma omp critical
            {
                batchCumulatedGrads.cg_hbias = batchCumulatedGrads.cg_hbias + threadCumulatedGrads.cg_hbias;
                batchCumulatedGrads.cg_Wi2h = batchCumulatedGrads.cg_Wi2h + threadCumulatedGrads.cg_Wi2h;
                batchCumulatedGrads.cg_Wh2o = batchCumulatedGrads.cg_Wh2o + threadCumulatedGrads.cg_Wh2o;
            }
        } // end multi-processor
        NNet<XPU>::UpdateCumulateGrads(batchCumulatedGrads, &netsParas);
    } // end total iteration
}
    
void Chunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cerr;
    using std::endl;

    cerr << "Training init..." << endl;
    cerr << "  Training Instance num: " << trainSet.size() << endl;

    m_featExtractor.reset(new FeatureExtractor());
    m_featExtractor->getDictionaries(goldSet);

    m_transitionSystem.reset(new ActionStandardSystem());
    m_transitionSystem->makeTransition(m_featExtractor->getKnownLabels());

#ifdef DEBUG
    m_transitionSystem->displayLabel2ActionIdx();
#endif

    m_fEmb.reset(new FeatureEmbedding(m_featExtractor->size(),
            CConfig::nFeatureNum,
            CConfig::nEmbeddingDim,
            m_nBeamSize));

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

#ifdef DEBUG1
    std::cerr << "train set size: " << trainSet.size() << std::endl;
    std::cerr << "dev gold set size: " << goldSet.size() << std::endl;
    std::cerr << "global examples size: " << gExamples.size() << std::endl;
#endif
}
