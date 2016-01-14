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
    
std::pair<BeamChunker::ChunkedResultType, BeamChunker::ChunkedResultType> BeamChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<XPU> &modelParas) {
    const int num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transSystemPtr->getActNumber();
    const int beam_size = m_nBeamSize;

    static int longestLen = (*std::max_element(devInstances.begin(), devInstances.end(), [](Instance &inst1, Instance &inst2){ return inst1.input.size() < inst2.input.size(); })).input.size();
    static const int nMaxLatticeSize = (m_nBeamSize + 1) * longestLen;
    int threads_num = CConfig::nThread;

    std::vector<State *> lattices(threads_num);
    std::vector<State **> lattice_indexes(threads_num);
    for (int i = 0; i < threads_num; i++) {
        lattices[i] = new State[nMaxLatticeSize];
        lattice_indexes[i] = new State*[longestLen + 2];
    }
    // std::vector<Model<XPU>> models;
    // for (int i = 0; i < threads_num; i++) {
    //     models.push_back(modelParas);
    // }

    static int chunkRound = 1;

    clock_t start, end;
    start = clock();
    ChunkedDataSet predictDevSet(goldDevSet.size());

    std::vector<ChunkedDataSet> threadPredictDevSets(threads_num);

#pragma omp parallel num_threads(threads_num)
    {
        int threadIndex = omp_get_thread_num();

        // TNNets tnnets(beam_size, num_in, num_hidden, num_out, &models[threadIndex], false);
        TNNets tnnets(beam_size, num_in, num_hidden, num_out, &modelParas, false);
        ChunkedDataSet &threadPredictDevSet = threadPredictDevSets[threadIndex];

        for (unsigned inst = threadIndex; inst < static_cast<unsigned>(devInstances.size()); inst += threads_num) {
            LabeledSequence predictSent(devInstances[inst].input);

            BeamDecoder decoder(&(devInstances[inst]), 
                                m_transSystemPtr,
                                m_featManagerPtr,
                                m_featEmbManagerPtr,
                                m_nBeamSize, 
                                lattices[threadIndex],
                                lattice_indexes[threadIndex],
                                false);

            decoder.generateLabeledSequence(tnnets, predictSent);

            threadPredictDevSet.push_back(predictSent);
        }
    }
    for (int i = 0; i < threads_num; i++) {
        delete []lattices[i];
        delete []lattice_indexes[i];
    }

    for (int i = 0; i < threads_num; i++) {
        for (int j = 0; j < threadPredictDevSets[i].size(); j++) {
            predictDevSet[i + j * threads_num] = threadPredictDevSets[i][j];
        }
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr << "[" << chunkRound << "] totally chunk " << devInstances.size() << " sentences, \ttime: " << time_used << " \taverage: " << devInstances.size() / time_used << " sentences/second!" << std::endl; chunkRound++;

    auto FB1 = Evalb::eval(predictDevSet, goldDevSet);
    auto NPFB1 = Evalb::eval(predictDevSet, goldDevSet, true);

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

    srand(0);

    InitTensorEngine<XPU>();
    Stream<XPU> *sstream = NewStream<XPU>();

    auto featureTypes = m_featManagerPtr->getFeatureTypes();
    std::cerr << "[begin]featureTypes:" << std::endl;
    for (auto &ft : featureTypes) {
        std::cerr << "  " << ft.typeName << ":" << std::endl;
        std::cerr << "    dictSize = " << ft.dictSize << std::endl;
        std::cerr << "    featSize = " << ft.featSize << std::endl;
        std::cerr << "    embsSize = " << ft.featEmbSize << std::endl;
    }
    std::cerr << "[end]" << std::endl;

    Model<XPU> modelParas(num_in, num_hidden, num_out, m_featEmbManagerPtr, sstream, true);
    Model<XPU> adaGradSquares(num_in, num_hidden, num_out, m_featEmbManagerPtr, sstream, false);

    auto longestSentence = *std::max_element(gExamples.begin(), gExamples.end(), [](GlobalExample &ge1, GlobalExample &ge2) { return ge1.instance.input.size() < ge2.instance.input.size();} );
    const int longestLen = longestSentence.instance.input.size();
    const int nMaxLatticeSize = (m_nBeamSize + 1) * longestLen;
    std::vector<State *> lattices(CConfig::nThread);
    std::vector<State **> lattice_indexes(CConfig::nThread);
    for (int i = 0; i < CConfig::nThread; i++) {
        lattices[i] = new State[nMaxLatticeSize];
        lattice_indexes[i] = new State*[longestLen + 2];
    }
    std::vector<std::vector<NNet<XPU> *>> netss;
    for (int i = 0; i < CConfig::nThread; i++) {
        std::vector<NNet<XPU> *> nets(longestLen + 1);

        for (int j = 0; j < nets.size(); j++) {
            nets[j] = new NNet<XPU>(m_nBeamSize, num_in, num_hidden, num_out, &modelParas);
        }

        netss.push_back(nets);
    }

    ChunkedResultType bestDevFB1 = std::make_tuple(0.0, 0.0, -1.0);
    ChunkedResultType bestDevNPFB1 = std::make_tuple(0.0, 0.0, -1.0);
    clock_t start, end;
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            auto res = chunk(devSet, devGoldSet, modelParas);
            ChunkedResultType &currentFB1 = std::get<0>(res);
            ChunkedResultType &currentNPFB1 = std::get<1>(res);

            if (std::get<2>(currentFB1) > std::get<2>(bestDevFB1)) {
                bestDevFB1 = currentFB1;
            }
            if (std::get<2>(currentNPFB1) > std::get<2>(bestDevNPFB1)) {
                bestDevNPFB1 = currentNPFB1;
            }
            auto sf = std::cerr.flags();
            auto sp = std::cerr.precision();
            std::cerr.flags(std::ios::fixed);
            std::cerr.precision(2);
            std::cerr << "current iteration FB1-score  : " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::get<0>(currentFB1) << "/" << std::get<1>(currentFB1) << "/" << std::get<2>(currentFB1) << "\t best FB1-score  : " << std::get<0>(bestDevFB1) << "/" << std::get<1>(bestDevFB1) << "/" << std::get<2>(bestDevFB1) << std::endl;
            std::cerr << "current iteration NPFB1-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::get<0>(currentNPFB1) << "/" << std::get<1>(currentNPFB1) << "/" << std::get<2>(currentNPFB1) << "\t best NPFB1-score: " << std::get<0>(bestDevNPFB1)  << "/" << std::get<1>(bestDevNPFB1) << "/" << std::get<2>(bestDevNPFB1) << std::endl;
            std::cerr.flags(sf);
            std::cerr.precision(sp);
        }

        start = clock();

        // random shuffle the training instances in the container,
        // and assign them for each thread
        std::vector<std::vector<GlobalExample *>> multiThread_miniBatch_data;
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

        Model<XPU> batchCumulatedGrads(num_in, num_hidden, num_out, m_featEmbManagerPtr, sstream, false);
        // begin to multi thread Training
#pragma omp parallel num_threads(CConfig::nThread)
        {
            int threadIndex = omp_get_thread_num();
            // int threadIndex = 0;
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];

            Model<XPU> cumulatedGrads(num_in, num_hidden, num_out, m_featEmbManagerPtr, sstream, false);

            // for evary instance in this mini-batch
            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                // fetch a to-be-trained instance
                GlobalExample *example = currentThreadData[inst];

                TNNets tnnets(m_nBeamSize, num_in, num_hidden, num_out, &modelParas, netss[threadIndex]);

                // decode and update
                // std::cerr << "begin to decode!" << std::endl;
                BeamDecoder decoder(&(example->instance), 
                                    m_transSystemPtr,
                                    m_featManagerPtr,
                                    m_featEmbManagerPtr,
                                    m_nBeamSize, 
                                    lattices[threadIndex],
                                    lattice_indexes[threadIndex],
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

            std::cerr << "[" << iter << "] totally train " << batch_size << " sentences, \ttime: " << time_used << " \taverage: " << batch_size / time_used << " sentences/second!" << std::endl; 
        }
    } // end total iteration

    for (int i = 0; i < CConfig::nThread; i++) {
        for (int j = 0; j < netss[i].size(); j++) {
            delete netss[i][j];
        }
    }
    for (int i = 0; i < CConfig::nThread; i++) {
        delete []lattices[i];
        delete []lattice_indexes[i];
    }
    DeleteStream(sstream);
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
        m_featManagerPtr,
                static_cast<real_t>(CConfig::fInitRange)
                ));

    m_transSystemPtr.reset(new ActionStandardSystem());
    m_transSystemPtr->init(goldSet);

    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), trainSet);

    GlobalExample::generateTrainingExamples(*(m_transSystemPtr.get()), *(m_featManagerPtr.get()), trainSet, goldSet, gExamples);

    std::cerr << std::endl << "  train set size: " << trainSet.size() << std::endl;
}
