/*************************************************************************
	> File Name: GreedyChunker.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 07 Dec 2015 08:56:14 PM CST
 ************************************************************************/
#include <ctime>
#include <omp.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits.h>

#include "Config.h"

#include "Evalb.h"

#include "GreedyChunker.h"

#define DEBUG

GreedyChunker::GreedyChunker() {

}

GreedyChunker::GreedyChunker(bool isTrain) {
    m_bTrain = isTrain;
}

GreedyChunker::~GreedyChunker() { } 

double GreedyChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsParas) {
    auto longestInst = *std::max_element(devInstances.begin(), devInstances.end(), [](Instance &inst1, Instance &inst2) { return inst1.size() < inst2.size();} );
    State *lattice = new State[longestInst.size() + 1];

    clock_t start, end;
    start = clock();
    ChunkedDataSet predDevSet;
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        Instance &currentInstance = devInstances[inst];
        predDevSet.push_back(ChunkedSentence(currentInstance.input));

        State* predState = decode(&currentInstance, netsParas, lattice);

        ChunkedSentence &predSent = predDevSet[inst];

        m_transitionSystem->generateOutput(*predState, predSent);
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr << "totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

    delete []lattice;

    auto res = Evalb::eval(predDevSet, goldDevSet);

    return std::get<2>(res);
}

void GreedyChunker::printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, NNetPara<XPU> &netsPara, double batchObjLoss, double posClassificationRate, double &bestDevFB1) {
    double currentFB1 = chunk(devSet, devGoldSet, netsPara);
    if (currentFB1 > bestDevFB1) {
        bestDevFB1 = currentFB1;
    }

    double loss = batchObjLoss;

    auto sf = std::cerr.flags();
    auto sp = std::cerr.precision();
    std::cerr.flags(std::ios::fixed);
    std::cerr.precision(2);
    std::cerr << "current iteration FB1-score: " << currentFB1 << "\tbest FB1-score: " << bestDevFB1 << std::endl;
    std::cerr << "current objective fun-score: " << loss << "\tclassfication rate: " << posClassificationRate << std::endl;
    std::cerr.flags(sf);
    std::cerr.precision(sp);
}

void GreedyChunker::generateMultiThreadsMiniBatchData(std::vector<ExamplePtrs> &multiThread_miniBatch_data) {
    int exampleNumOfThread = std::min(CConfig::nBatchSize, static_cast<int>(trainExamplePtrs.size())) / CConfig::nThread;

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

void display1Tensor( Tensor<cpu, 1, real_t> & tensor ){
    for(int i = 0; i < tensor.size(0); i++)
        std::cerr<<tensor[i]<<" ";
    std::cerr<<std::endl;
}

void display2Tensor( Tensor<cpu, 2, double> tensor ){
    std::cerr<<"size 0 :" << tensor.size(0)<<" size 1: "<<tensor.size(1)<<std::endl;
    for(int i = 0; i < tensor.size(0); i++){
       for(int j = 0; j < tensor.size(1); j++)
           std::cerr<<tensor[i][j]<<" ";
       std::cerr<<std::endl;
    }
}

void GreedyChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cerr << "Initing FeatureManager & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cerr << "Excuting devset generateInstanceSetCache & readPretrainEmbeddings..." << std::endl;
    m_featManager->generateInstanceSetCache(devSet);
    std::cerr << "  Greedy train set size: " << trainExamplePtrs.size() << std::endl;
    m_featManager->readPretrainEmbeddings(CConfig::strEmbeddingPath);

    const static int num_in = m_featManager->totalFeatSize;
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transitionSystem->nActNum;
    const static int batchSize = std::min(CConfig::nBatchSize, static_cast<int>(trainExamplePtrs.size()));

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    InitTensorEngine<XPU>();

    NNetPara<XPU> netsParas(1, num_in, num_hidden, num_out);

    double bestDevFB1 = -1.0;

    int batchCorrectSize = 0;
    double batchObjLoss = 0.0;

    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double posClassificationRate = 100 * static_cast<double>(batchCorrectSize) / batchSize;

            printEvaluationInfor(devSet, devGoldSet, netsParas, batchObjLoss, posClassificationRate, bestDevFB1);
        }
        batchCorrectSize = 0;
        batchObjLoss = 0.0;

        // random shuffle the training instances in the container,
        // and assign them for each threads
        std::vector<ExamplePtrs> multiThread_miniBatch_data;

        // prepare mini-batch data for each threads
        std::random_shuffle(trainExamplePtrs.begin(), trainExamplePtrs.end());
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);
        UpdateGrads<XPU> batchCumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);


#pragma omp parallel
        {
            int threadIndex = omp_get_thread_num();
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];

            int threadCorrectSize = 0;
            double threadObjLoss = 0.0;

            UpdateGrads<XPU> cumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);
            std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &netsParas));

            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {

                Example *e = currentThreadData[inst];

                TensorContainer<cpu, 2, real_t> input;
                input.Resize(Shape2(1, num_in));

                TensorContainer<cpu, 2, real_t> pred;
                pred.Resize(Shape2(1, num_out));

                std::vector<FeatureVector> featureVectors;
                featureVectors.push_back(e->features);
                m_featManager->returnInput(featureVectors, input, 1);

                nnet->Forward(input, pred, false);

                std::vector<int> validActs(e->labels);

                int optAct = -1;
                int goldAct = -1;
                for (int i = 0; i < validActs.size(); i++) {
                    if (validActs[i] >= 0) {
                        if (optAct == -1 || pred[0][i] > pred[0][optAct]){
                            optAct = i;
                        }

                        if (validActs[i] == 1) {
                            goldAct = i;
                        }
                    }
                }
                if (optAct == goldAct) {
                    threadCorrectSize += 1;
                }

                real_t maxScore = pred[0][optAct];
                real_t goldScore = pred[0][goldAct];

                real_t sum = 0.0;
                for (int i = 0; i < validActs.size(); i++) {
                    if (validActs[i] >= 0) {
                        pred[0][i] = std::exp(pred[0][i] - maxScore);
                        sum += pred[0][i];
                    }
                }

                threadObjLoss += (std::log(sum) - (goldScore - maxScore)) / batchSize;

                for (int i = 0; i < validActs.size(); i++) {
                    if (validActs[i] >= 0) {
                        pred[0][i] = pred[0][i] / sum;
                    } else {
                        pred[0][i] = 0.0;
                    }
                }
                pred[0][goldAct] -= 1.0;

                for (int i = 0; i < validActs.size(); i++) {
                    pred[0][i] /= batchSize;
                }

                nnet->Backprop(pred);
                nnet->SubsideGrads(cumulatedGrads);
            }

#pragma omp barrier
#pragma omp critical 
            {
                batchCumulatedGrads.cg_hbias = batchCumulatedGrads.cg_hbias + cumulatedGrads.cg_hbias;
                batchCumulatedGrads.cg_Wi2h = batchCumulatedGrads.cg_Wi2h + cumulatedGrads.cg_Wi2h;
                batchCumulatedGrads.cg_Wh2o = batchCumulatedGrads.cg_Wh2o + cumulatedGrads.cg_Wh2o;
            }

#pragma omp critical 
            batchCorrectSize += threadCorrectSize;

#pragma omp critical 
            batchObjLoss += threadObjLoss;
        
        }  // end multi-processor

        NNet<XPU>::UpdateCumulateGrads(batchCumulatedGrads, &netsParas);
    }

    ShutdownTensorEngine<XPU>();
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cerr;
    using std::endl;

    m_featManager.reset(new FeatureManager());
    m_featManager->init(goldSet, CConfig::fInitRange);

    m_transitionSystem.reset(new ActionStandardSystem());
    m_transitionSystem->makeTransition(m_featManager->getKnownLabels());

    m_featManager->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

    for (auto &gExample : gExamples) {
        for (auto &example : gExample.examples) {
            trainExamplePtrs.push_back(&(example));
        }
    }
}

State* GreedyChunker::decode(Instance *inst, NNetPara<XPU> &paras, State *lattice) {
    const static int num_in = m_featManager->totalFeatSize;
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transitionSystem->nActNum;

    int nSentLen = inst->input.size();
    int nMaxRound = nSentLen;
    FeatureManager &fManager = *(m_featManager.get());
    ActionStandardSystem &tranSystem = *(m_transitionSystem.get());
    std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &paras));

    State *retval = nullptr;
    for (int i = 0; i < nMaxRound + 1; ++i) {
        lattice[i].m_nLen = nSentLen;
    }

    lattice[0].clear();

    InitTensorEngine<XPU>();
    for (int nRound = 1; nRound <= nMaxRound; nRound++){
        State *currentState = lattice + nRound - 1;
        State *target = lattice + nRound;

        TensorContainer<cpu, 2, real_t> input;
        input.Resize(Shape2(1, num_in));

        TensorContainer<cpu, 2, real_t> pred;
        pred.Resize(Shape2(1, num_out));
       
        std::vector<FeatureVector> featureVectors;
        featureVectors.push_back(FeatureVector(fManager.featTypes, fManager.featEmbs));
        generateInputBatch(currentState, inst, featureVectors);
        fManager.returnInput(featureVectors, input, 1);

        nnet->Forward(input, pred, false);
        
        std::vector<int> validActs;
        tranSystem.generateValidActs(*currentState, validActs);
        // get max-score valid action
        real_t maxScore = 0.0;
        unsigned maxActID = 0;
        
        for (unsigned actID = 0; actID < validActs.size(); ++actID) {
            if (validActs[actID] == -1) {
                continue;
            }

            if (actID == 0 || pred[0][actID] > maxScore) {
                maxScore = pred[0][actID];
                maxActID = actID;
            }
        }

        CScoredTransition trans(currentState, maxActID, currentState->score + maxScore);
        *target = *currentState;
        tranSystem.move(*currentState, *target, trans);
        retval = target;
    }

    ShutdownTensorEngine<XPU>();

    return retval;
}

void GreedyChunker::generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs) {
    for (int i = 0; i < featvecs.size(); i++) {
        m_featManager->extractFeature(*(state + i), *inst, featvecs[i]);
    }
}
