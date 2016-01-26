/*************************************************************************
	> File Name: GreedyChunker.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 07 Dec 2015 08:56:14 PM CST
 ************************************************************************/
#include <chrono>
#include <omp.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits.h>

#include "Config.h"

#include "Evalb.h"

#include "GreedyChunker.h"

#include "GreedyChunkerThread.h"

const double MICROSECOND = 1000000.0;

GreedyChunker::GreedyChunker() {

}

GreedyChunker::GreedyChunker(bool isTrain) {
    m_bTrain = isTrain;
}

GreedyChunker::~GreedyChunker() { } 

std::pair<GreedyChunker::ChunkedResultType, GreedyChunker::ChunkedResultType> GreedyChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<cpu> &modelParas) {
    int threads_num = CConfig::nThread;

    static int chunkRound = 1;

    auto start = std::chrono::high_resolution_clock::now();

    ChunkedDataSet predictDevSet(goldDevSet.size());

    std::vector<ChunkedDataSet> threadPredictDevSets(threads_num);

#pragma omp parallel num_threads(threads_num)
    {
        int threadIndex =  omp_get_thread_num();
        SetDevice<gpu>(threadIndex);

        m_chunkerThreadPtrs[threadIndex]->chunk(threads_num, modelParas, devInstances, threadPredictDevSets[threadIndex]);
#pragma omp barrier
    }

    for (int i = 0; i < threads_num; i++) {
        for (int j = 0; j < static_cast<int>(threadPredictDevSets[i].size()); j++) {
            predictDevSet[i + j * threads_num] = threadPredictDevSets[i][j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MICROSECOND;
    std::cerr << "[" << chunkRound << "] totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl; chunkRound++;

    ChunkedResultType FB1 = Evalb::eval(predictDevSet, goldDevSet);
    ChunkedResultType NPFB1 = Evalb::eval(predictDevSet, goldDevSet, true);

    return std::make_pair(FB1, NPFB1);
}

void GreedyChunker::printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, Model<cpu> &modelParas, double batchObjLoss, double posClassificationRate, ChunkedResultType &bestDevFB1, ChunkedResultType &bestDevNPFB1) {
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

    double loss = batchObjLoss;

    auto sf = std::cerr.flags();
    auto sp = std::cerr.precision();
    std::cerr.flags(std::ios::fixed);
    std::cerr.precision(2);
    std::cerr << "current iteration FB1-score  : " << std::get<0>(currentFB1) << "/" << std::get<1>(currentFB1) << "/" << std::get<2>(currentFB1) << "\tbest FB1-score  : " << std::get<0>(bestDevFB1) << "/" << std::get<1>(bestDevFB1) << "/" << std::get<2>(bestDevFB1) << std::endl;
    std::cerr << "current iteration NPFB1-score: " << std::get<0>(currentNPFB1) << "/" << std::get<1>(currentNPFB1) << "/" << std::get<2>(currentNPFB1) << "\tbest NPFB1-score: " << std::get<0>(bestDevNPFB1) << "/" << std::get<1>(bestDevNPFB1) << "/" << std::get<2>(bestDevNPFB1) << std::endl;
    std::cerr << "current objective fun-score  : " << loss << "\tclassfication rate: " << posClassificationRate << std::endl;
    std::cerr.flags(sf);
    std::cerr.precision(sp);
}

void GreedyChunker::generateMultiThreadsMiniBatchData(std::vector<ExamplePtrs> &multiThread_miniBatch_data) {
    std::random_shuffle(trainExamplePtrs.begin(), trainExamplePtrs.end());

    static int exampleNumOfThread = std::min(CConfig::nGreedyBatchSize, static_cast<int>(trainExamplePtrs.size())) / CConfig::nThread;

    auto sp = trainExamplePtrs.begin();
    auto ep = sp + exampleNumOfThread;
    for (int i = 0; i < CConfig::nThread; i++) {
        ExamplePtrs threadExamples;

        for (auto p = sp; p != ep; p++) {
            threadExamples.push_back(*p);
        }

        multiThread_miniBatch_data.push_back(threadExamples);

        sp = ep;
        ep += exampleNumOfThread;
    }
}

void GreedyChunker::saveChunker(int round) {
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

void GreedyChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cerr << "[train involved]Initing DictManager &  FeatureManager & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cerr << "[dev involved]Initing generateInstanceSetCache for devSet..." << std::endl;
    initDev(devSet);

    std::cerr << "[chunkthreads involved]Initing chunkerthreads..." << std::endl;
    initGreedyChunkerThread(devSet);

    Model<cpu> &modelParas = *(m_modelPtr.get());
    auto featureTypes = m_featManagerPtr->getFeatureTypes();
    Model<cpu> adaGradSquares(num_in, num_hidden, num_out, featureTypes, NULL);

    ChunkedResultType bestDevFB1 = std::make_tuple(0.0, 0.0, -1.0);
    ChunkedResultType bestDevNPFB1 = std::make_tuple(0.0, 0.0, -1.0);

    const int batchSize = std::min(CConfig::nGreedyBatchSize, static_cast<int>(trainExamplePtrs.size()));
    int batchCorrectSize = 0;
    double batchObjLoss = 0.0;
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (CConfig::saveModel && iter % CConfig::nSaveModelPerIters == 0) {
            saveChunker(iter);
        }
        if (iter % CConfig::nEvaluatePerIters == 0) {

            double posClassificationRate = 100 * static_cast<double>(batchCorrectSize) / batchSize;

            double regular_loss = 0.5 * CConfig::fRegularizationRate * modelParas.norm2();
            printEvaluationInfor(devSet, devGoldSet, modelParas, batchObjLoss + regular_loss, posClassificationRate, bestDevFB1, bestDevNPFB1);
            // std::cerr << "Regularization loss: " << regular_loss << std::endl;
            // printEvaluationInfor(devSet, devGoldSet, modelParas, batchObjLoss, posClassificationRate, bestDevFB1, bestDevNPFB1);
        }
        batchCorrectSize = 0;
        batchObjLoss = 0.0;

        auto start = std::chrono::high_resolution_clock::now();

        // random shuffle the training instances in the container,
        // and assign them for each threads
        std::vector<ExamplePtrs> multiThread_miniBatch_data;
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

        Model<cpu> batchCumulatedGrads(num_in, num_hidden, num_out, featureTypes, NULL);
        
#pragma omp parallel num_threads(CConfig::nThread)
        {
            int threadIndex = omp_get_thread_num();
            SetDevice<gpu>(threadIndex);

            int threadCorrectSize = 0;
            double threadObjLoss = 0.0;

            Model<cpu> cumulatedGrads(num_in, num_hidden, num_out, featureTypes, NULL);

            auto currentThreadData = multiThread_miniBatch_data[threadIndex];
            m_chunkerThreadPtrs[threadIndex]->train(modelParas, currentThreadData, batchSize, cumulatedGrads, threadCorrectSize, threadObjLoss);

#pragma omp barrier
#pragma omp critical 
            batchCumulatedGrads.mergeModel(&cumulatedGrads);

#pragma omp critical 
            batchCorrectSize += threadCorrectSize;
#pragma omp critical 
            batchObjLoss += threadObjLoss;
        
        }  // end multi-processor

        modelParas.update(&batchCumulatedGrads, &adaGradSquares);

        auto end = std::chrono::high_resolution_clock::now();
        if (iter % CConfig::nEvaluatePerIters == 0) 
        {
            double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / MICROSECOND;
            std::cerr << "[" << iter << "] totally train " << batchSize << " examples, time: " << time_used << " average: " << batchSize / time_used << " examples/second!" << std::endl; 
        }
    }
}

void GreedyChunker::initGreedyChunkerThread(InstanceSet &devSet) {
    const int longestLen = (*std::max_element(devSet.begin(), 
                devSet.end(), 
                [](Instance &inst1, Instance &inst2) { 
                    return inst1.size() < inst2.size();
                })
            ).input.size();

    std::cerr << "  longest sentence size: " << longestLen << std::endl;
    
    m_chunkerThreadPtrs.resize(CConfig::nThread);
    for (int i = 0; i < CConfig::nThread; i++) {
        m_chunkerThreadPtrs[i].reset(new GreedyChunkerThread(i, CConfig::nGPUBatchSize, *(m_modelPtr.get()), m_transSystemPtr, m_featManagerPtr, m_featEmbManagerPtr, longestLen));
    }
}

void GreedyChunker::initDev(InstanceSet &devSet) {
    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), devSet);
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
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

    for (auto &gExample : gExamples) {
        for (auto &example : gExample.examples) {
            trainExamplePtrs.push_back(&(example));
        }
    }

    auto featureTypes = m_featManagerPtr->getFeatureTypes();

    std::cerr << "  total input embedding dim: " << m_featEmbManagerPtr->getTotalFeatEmbSize() << std::endl;
    std::cerr << "  greedy train set size: " << trainExamplePtrs.size() << std::endl;
    std::cerr << "  [begin]featureTypes:" << std::endl;
    for (auto &ft : featureTypes) {
        std::cerr << "    " << ft.typeName << ":" << std::endl;
        std::cerr << "      dictSize = " << ft.dictSize << std::endl;
        std::cerr << "      featSize = " << ft.featSize << std::endl;
        std::cerr << "      embsSize = " << ft.featEmbSize << std::endl;
    }
    std::cerr << "  [end]" << std::endl;
}

