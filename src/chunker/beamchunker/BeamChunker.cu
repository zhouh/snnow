/*************************************************************************
	> File Name: BeamChunker.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 03:59:17 PM CST
 ************************************************************************/
#include <chrono>
#include <omp.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits.h>

#include "Config.h"

#include "TNNets.h"
#include "BeamChunker.h"
#include "BatchBeamDecoder.h"
#include "Evalb.h"

#include "Example.h"

#include "BeamChunkerThread.h"

const double MICROSECOND = 1000000.0;

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
    
std::pair<BeamChunker::ChunkedResultType, BeamChunker::ChunkedResultType> BeamChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<cpu> &modelParas) {
    int threads_num = CConfig::nThread;

    static int chunkRound = 1;

    auto start = std::chrono::high_resolution_clock::now();

    ChunkedDataSet predictDevSet(goldDevSet.size());

    std::vector<ChunkedDataSet> threadPredictDevSets(threads_num);

#pragma omp parallel num_threads(threads_num)
    {
        int threadIndex = omp_get_thread_num();
        SetDevice<XPU>(threadIndex);

        m_chunkerThreadPtrs[threadIndex]->chunk(threads_num, modelParas, devInstances, threadPredictDevSets[threadIndex]);
    }

    for (int i = 0; i < threads_num; i++) {
        for (int j = 0; j < threadPredictDevSets[i].size(); j++) {
            predictDevSet[i + j * threads_num] = threadPredictDevSets[i][j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MICROSECOND;
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

void BeamChunker::initBeamChunkerThread(InstanceSet &devSet) {
    const int trainLongestLen = (*std::max_element(gExamples.begin(), 
                gExamples.end(), 
                [](GlobalExample &ge1, GlobalExample &ge2) { 
                    return ge1.instance.input.size() < ge2.instance.input.size();
                })
            ).instance.input.size();
    static int devLongestLen = (*std::max_element(devSet.begin(), 
                devSet.end(), 
                [](Instance &inst1, Instance &inst2){ 
                    return inst1.input.size() < inst2.input.size(); 
                })
            ).input.size();

    const int longestLen = std::max(trainLongestLen, devLongestLen);
    std::cerr << "  longest sentence size: " << longestLen << std::endl;

    m_chunkerThreadPtrs.resize(CConfig::nThread);
    for (int i = 0; i < CConfig::nThread; i++) {
        m_chunkerThreadPtrs[i].reset(new BeamChunkerThread(i, m_nBeamSize, *(m_modelPtr.get()), m_transSystemPtr, m_featManagerPtr, m_featEmbManagerPtr, longestLen));
    }
}

void BeamChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {

    std::cerr << "[train involved]Initing DictManager &  FeatureManager & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cerr << "[dev involved]Initing generateInstanceSetCache for devSet..." << std::endl;
    initDev(devSet);

    std::cerr << "[chunkthreads involved]Initing chunkerthreads ..." << std::endl;
    initBeamChunkerThread(devSet);

    Model<cpu> &modelParas = *(m_modelPtr.get());
    auto featureTypes = m_featManagerPtr->getFeatureTypes();
    Model<cpu> adaGradSquares(num_in, num_hidden, num_out, featureTypes, NULL);

    ChunkedResultType bestDevFB1 = std::make_tuple(0.0, 0.0, -1.0);
    ChunkedResultType bestDevNPFB1 = std::make_tuple(0.0, 0.0, -1.0);

    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (CConfig::saveModel && iter % CConfig::nSaveModelPerIters == 0) {
            saveChunker(iter);
        }
        if (iter % CConfig::nEvaluatePerIters == 0) {
            auto res = chunk(devSet, devGoldSet, modelParas);
            ChunkedResultType &currentFB1 = std::get<0>(res);
            ChunkedResultType &currentNPFB1 = std::get<1>(res);

            if (std::get<2>(currentFB1) > std::get<2>(bestDevFB1)) {
                bestDevFB1 = currentFB1;
                bestDevNPFB1 = currentNPFB1;
                if (CConfig::saveModel) {
                    saveChunker(0);
                }
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

        auto start = std::chrono::high_resolution_clock::now();

        // random shuffle the training instances in the container, and assign them for each thread
        std::vector<std::vector<GlobalExample *>> multiThread_miniBatch_data;
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

        Model<cpu> batchCumulatedGrads(num_in, num_hidden, num_out, featureTypes, NULL);

        // begin to multi thread Training
#pragma omp parallel num_threads(CConfig::nThread)
        {
            int threadIndex = omp_get_thread_num();
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];

            Model<cpu> cumulatedGrads(num_in, num_hidden, num_out, featureTypes, NULL);

            SetDevice<gpu>(threadIndex);
            m_chunkerThreadPtrs[threadIndex]->train(modelParas, currentThreadData, cumulatedGrads);

#pragma omp barrier
#pragma omp critical
            batchCumulatedGrads.mergeModel(&cumulatedGrads);

        } // end multi-processor

        modelParas.update(&batchCumulatedGrads, &adaGradSquares);

        auto end = std::chrono::high_resolution_clock::now();
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MICROSECOND;

            static const int batch_size = std::min(CConfig::nBeamBatchSize, static_cast<int>(gExamples.size()));
            std::cerr << "[" << iter << "] totally train " << batch_size << " sentences, \ttime: " << time_used << " \taverage: " << batch_size / time_used << " sentences/second!" << std::endl; 
        }
    } // end total iteration

}

void BeamChunker::initDev(InstanceSet &devSet) {
    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), devSet);
}

void BeamChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cerr;
    using std::endl;

    m_dictManagerPtr.reset(new DictManager());
    m_featManagerPtr.reset(new FeatureManager());
    m_featEmbManagerPtr.reset(new FeatureEmbeddingManager());
    m_transSystemPtr.reset(new ActionStandardSystem());
    if (CConfig::loadModel){
        std::ifstream dict_is(CConfig::strModelDirPath + "/dictionarymanager.model");
        m_dictManagerPtr->loadDictManager(dict_is);

        std::ifstream featManager_is(CConfig::strModelDirPath + "/featuremanager.model");
        m_featManagerPtr->loadFeatureManager(featManager_is, m_dictManagerPtr);

        std::ifstream trans_is(CConfig::strModelDirPath + "/actionsystem.model");
        m_transSystemPtr->loadActionSystem(trans_is);
    } else {
        m_dictManagerPtr->init(goldSet);
        m_featManagerPtr->init(goldSet, m_dictManagerPtr);
        m_transSystemPtr->init(goldSet);
    }

    m_featEmbManagerPtr->init(m_featManagerPtr);

    num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    num_hidden = CConfig::nHiddenSize;
    num_out = m_transSystemPtr->getActNumber();

    srand(0);

    // Stream<XPU> *sstream = NewStream<XPU>();

    m_modelPtr.reset(new Model<cpu>(num_in, num_hidden, num_out, m_featEmbManagerPtr->getFeatureTypes(), NULL));
    if (CConfig::loadModel) {
        std::ifstream model_is(CConfig::strModelDirPath + "/netmodel.model");
        m_modelPtr->loadModel(model_is);
    } else {
        m_modelPtr->randomInitialize();
    }

    if (!CConfig::loadModel && CConfig::bReadPretrain) {
        m_featEmbManagerPtr->readPretrainedEmbeddings(*(m_modelPtr.get()));
    }

    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), trainSet);

    GlobalExample::generateTrainingExamples(*(m_transSystemPtr.get()), *(m_featManagerPtr.get()), trainSet, goldSet, gExamples);

    auto featureTypes = m_featManagerPtr->getFeatureTypes();

    std::cerr << "  total input embedding dim: " << m_featEmbManagerPtr->getTotalFeatEmbSize() << std::endl;
    std::cerr << std::endl << "  train set size: " << trainSet.size() << std::endl;
    std::cerr << "  [begin]featureTypes:" << std::endl;
    for (auto &ft : featureTypes) {
        std::cerr << "    " << ft.typeName << ":" << std::endl;
        std::cerr << "      dictSize = " << ft.dictSize << std::endl;
        std::cerr << "      featSize = " << ft.featSize << std::endl;
        std::cerr << "      embsSize = " << ft.featEmbSize << std::endl;
    }
    std::cerr << "  [end]" << std::endl;
}

void BeamChunker::saveChunker(int round) {
    std::string dir = CConfig::strModelDirPath;
    std::string app_str;

    if (round != -1) {
        app_str = "." + std::to_string(round);
    }

    std::ofstream actionSystemOs(dir + "/actionsystem.model" + app_str);
    m_transSystemPtr->saveActionSystem(actionSystemOs);

    std::ofstream dictOs(dir + "/dictionarymanager.model" + app_str);
    m_dictManagerPtr->saveDictManager(dictOs);

    std::ofstream featManagerOs(dir + "/featuremanager.model" + app_str);
    m_featManagerPtr->saveFeatureManager(featManagerOs);

    std::ofstream modelOs(dir + "/netmodel.model" + app_str);
    m_modelPtr->saveModel(modelOs);
}
