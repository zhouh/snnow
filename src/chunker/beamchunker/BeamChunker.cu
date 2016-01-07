/*************************************************************************
	> File Name: BeamChunker.cpp
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

#include "TNNets.h"
#include "BeamChunker.h"
#include "Evalb.h"

#include "Example.h"

BeamChunker::BeamChunker() {
    m_nBeamSize = CConfig::nBeamSize;
    m_bTrain = false;
}

BeamChunker::BeamChunker(bool isTrain) {
    m_nBeamSize = CConfig::nBeamSize;
    m_bTrain = isTrain;
}

BeamChunker::~BeamChunker() {
}
    
std::pair<double, double> BeamChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<XPU> &modelParas) {
    const int num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transSystemPtr->getActNumber();
    const int beam_size = CConfig::nBeamSize;
    static int chunkRound = 1;

    TNNets tnnets(beam_size, num_in, num_hidden, num_out, &modelParas, false);

    clock_t start, end;
    start = clock();
    ChunkedDataSet predictDevSet;
    for (unsigned inst = 0; inst < static_cast<unsigned>(devInstances.size()); inst++) {
        LabeledSequence predictSent(devInstances[inst].input);

        BeamDecoder decoder(&(devInstances[inst]), 
                            m_transSystemPtr,
                            m_featManagerPtr,
                            m_featEmbManagerPtr,
                            m_nBeamSize, 
                            false);

        decoder.generateLabeledSequence(tnnets, predictSent);

        predictDevSet.push_back(predictSent);
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr << "[" << chunkRound << "] totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl; chunkRound++;

    auto res = Evalb::eval(predictDevSet, goldDevSet);
    double FB1 = std::get<2>(res);
    res = Evalb::eval(predictDevSet, goldDevSet);
    double NPFB1 = std::get<2>(res);

    return std::make_pair(FB1, NPFB1);
}

void BeamChunker::generateMultiThreadsMiniBatchData(std::vector<std::vector<GlobalExample *>> &multiThread_miniBatch_data) {
    std::random_shuffle(gExamples.begin(), gExamples.end());

    // prepare mini-batch data for each threads
    static int exampleNumOfThread = std::min(CConfig::nBeamBatchSize, static_cast<int>(gExamples.size()))/ CConfig::nThread;

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
}

void BeamChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cerr << "[trainingSet involved initing]Initing DictManager &  FeatureManager & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cerr << "[devSet involved initing]Initing generateInstanceSetCache for devSet..." << std::endl;
    initDev(devSet);

    const int num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transSystemPtr->getActNumber();
    const int batch_size = std::min(CConfig::nBeamBatchSize, static_cast<int>(gExamples.size()));

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    InitTensorEngine<XPU>();
    std::vector<Stream<XPU> *> streams(CConfig::nThread + 1);

    auto featureTypes = m_featManagerPtr->getFeatureTypes();
    std::cerr << "[begin]featureTypes:" << std::endl;
    for (auto &ft : featureTypes) {
        std::cerr << "  " << ft.typeName << ":" << std::endl;
        std::cerr << "    dictSize = " << ft.dictSize << std::endl;
        std::cerr << "    featSize = " << ft.featSize << std::endl;
        std::cerr << "    embsSize = " << ft.featEmbSize << std::endl;
    }
    std::cerr << "[end]" << std::endl;

    Model<XPU> modelParas(num_in, num_hidden, num_out, featureTypes, streams[0], true);
    m_featEmbManagerPtr->readPretrainedEmbeddings(modelParas);
    Model<XPU> adaGradSquares(num_in, num_hidden, num_out, featureTypes, streams[0], false);

    double bestDevFB1 = -1.0;
    double bestDevNPFB1 = -1.0;
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            auto res = chunk(devSet, devGoldSet, modelParas);
            double currentFB1 = std::get<0>(res);
            double currentNPFB1 = std::get<1>(res);

            if (currentFB1 > bestDevFB1) {
                bestDevFB1 = currentFB1;
            }
            if (currentNPFB1 > bestDevNPFB1) {
                bestDevNPFB1 = currentNPFB1;
            }
            auto sf = std::cerr.flags();
            auto sp = std::cerr.precision();
            std::cerr.flags(std::ios::fixed);
            std::cerr.precision(2);
            std::cerr << "current iteration FB1-score  : " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << currentFB1 << "\t   best FB1-score: " << bestDevFB1 << std::endl;
            std::cerr << "current iteration NPFB1-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << currentNPFB1 << "\t best NPFB1-score: " << bestDevNPFB1 << std::endl;
            std::cerr.flags(sf);
            std::cerr.precision(sp);
        }

        clock_t start, end;
        start = clock();

        // random shuffle the training instances in the container,
        // and assign them for each thread
        std::vector<std::vector<GlobalExample *>> multiThread_miniBatch_data;
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

        Model<XPU> batchCumulatedGrads(num_in, num_hidden, num_out, featureTypes, streams[0], false);
        // begin to multi thread Training
#pragma omp parallel
        {
            int threadIndex = omp_get_thread_num();
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];

            Model<XPU> cumulatedGrads(num_in, num_hidden, num_out, featureTypes, streams[threadIndex + 1], false);

            // for evary instance in this mini-batch
            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                // fetch a to-be-trained instance
                GlobalExample *example = currentThreadData[inst];

                TNNets tnnets(m_nBeamSize, num_in, num_hidden, num_out, &modelParas);

                // decode and update
                // std::cerr << "begin to decode!" << std::endl;
                BeamDecoder decoder(&(example->instance), 
                                    m_transSystemPtr,
                                    m_featManagerPtr,
                                    m_featEmbManagerPtr,
                                    m_nBeamSize, 
                                    true);

                State * predState = decoder.decode(tnnets, example);

                tnnets.updateTNNetParas(&cumulatedGrads, decoder.beam, decoder.bEarlyUpdate, decoder.nGoldTransitionIndex, decoder.goldScoredTran);
            } // end for instance traverse

#pragma omp barrier
#pragma omp critical
            batchCumulatedGrads.mergeModel(&cumulatedGrads);
        } // end multi-processor

        modelParas.update(&batchCumulatedGrads, &adaGradSquares);

        end = clock();
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double time_used = (double)(end - start) / CLOCKS_PER_SEC;

            std::cerr << "[" << iter << "] totally train " << batch_size << " sentences, time: " << time_used << " average: " << batch_size / time_used << " sentences/second!" << std::endl; 
        }
    } // end total iteration

    ShutdownTensorEngine<XPU>();
}

void BeamChunker::initDev(InstanceSet &devSet) {
    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), devSet);
}

void BeamChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cerr;
    using std::endl;

    m_dictManagerPtr.reset(new DictManager());
    m_dictManagerPtr->init(goldSet);

    m_featManagerPtr.reset(new FeatureManager());
    m_featManagerPtr->init(goldSet, m_dictManagerPtr);

    m_featEmbManagerPtr.reset(new FeatureEmbeddingManager(
                m_featManagerPtr->getFeatureTypes(),
                m_featManagerPtr->getDictManagerPtrs(),
                static_cast<real_t>(CConfig::fInitRange)
                ));

    m_transSystemPtr.reset(new ActionStandardSystem());
    m_transSystemPtr->init(goldSet);

    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), trainSet);

    GlobalExample::generateTrainingExamples(*(m_transSystemPtr.get()), *(m_featManagerPtr.get()), trainSet, goldSet, gExamples);

    std::cerr << std::endl << "  train set size: " << trainSet.size() << std::endl;
}
