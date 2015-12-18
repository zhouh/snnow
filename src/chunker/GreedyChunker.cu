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

#ifdef DEBUG
#define ADDREGURLOSS
// #define CLOSEOPENOMP
#endif

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
    std::cout << "totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

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
#ifdef ADDREGURLOSS
    double paraLoss = 0.0;
    for (int ii = 0; ii < netsPara.Wi2h.shape_[0]; ii++) {
        for (int jj = 0; jj < netsPara.Wi2h.shape_[1]; jj++) {
            paraLoss += netsPara.Wi2h[ii][jj] * netsPara.Wi2h[ii][jj];
        }
    }
    for (int ii = 0; ii < netsPara.Wh2o.shape_[0]; ii++) {
        for (int jj = 0; jj < netsPara.Wh2o.shape_[1]; jj++) {
            paraLoss += netsPara.Wh2o[ii][jj] * netsPara.Wh2o[ii][jj];
        }
    }
    for (int ii = 0; ii < netsPara.hbias.shape_[0]; ii++) {
        paraLoss += netsPara.hbias[ii] * netsPara.hbias[ii];
    }
    std::cout << "current |W|^2: " << paraLoss << std::endl;
    paraLoss *= 0.5 * CConfig::fRegularizationRate;

    loss += paraLoss;
#endif

    auto sf = std::cout.flags();
    auto sp = std::cout.precision();
    std::cout.flags(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "current iteration FB1-score: " << currentFB1 << "\tbest FB1-score: " << bestDevFB1 << std::endl;
    std::cout << "current objective fun-score: " << loss << "\tclassfication rate: " << posClassificationRate << std::endl;
    std::cout.flags(sf);
    std::cout.precision(sp);
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
        std::cout<<tensor[i]<<" ";
    std::cout<<std::endl;
}

void display2Tensor( Tensor<cpu, 2, double> tensor ){
    std::cout<<"size 0 :" << tensor.size(0)<<" size 1: "<<tensor.size(1)<<std::endl;
    for(int i = 0; i < tensor.size(0); i++){
       for(int j = 0; j < tensor.size(1); j++)
           std::cout<<tensor[i][j]<<" ";
       std::cout<<std::endl;
    }
}

void GreedyChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cout << "Initing FeatureExtractor & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cout << "Excuting generateInstanceSetCache & readPretrainEmbeddings..." << std::endl;
    m_featExtractor->generateInstanceSetCache(devSet);

    // m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);

    const static int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
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
        // std::random_shuffle(trainExamplePtrs.begin(), trainExamplePtrs.end());
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);
        UpdateGrads<XPU> batchCumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);


#ifndef CLOSEOPENOMP 
#pragma omp parallel
#endif
        {
            int threadIndex = omp_get_thread_num();
#ifndef CLOSEOPENOMP
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];
#endif
#ifdef CLOSEOPENOMP
            auto currentThreadData = multiThread_miniBatch_data[0];
#endif
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

                std::vector<std::vector<int>> featureVectors;
                featureVectors.push_back(e->features);
                m_fEmb->returnInput(featureVectors, input);

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

#ifndef CLOSEOPENOMP 
#pragma omp barrier
#pragma omp critical 
#endif            
            {
                batchCumulatedGrads.cg_hbias = batchCumulatedGrads.cg_hbias + cumulatedGrads.cg_hbias;
                batchCumulatedGrads.cg_Wi2h = batchCumulatedGrads.cg_Wi2h + cumulatedGrads.cg_Wi2h;
                batchCumulatedGrads.cg_Wh2o = batchCumulatedGrads.cg_Wh2o + cumulatedGrads.cg_Wh2o;
            }

#ifndef CLOSEOPENOMP 
#pragma omp critical 
#endif
            batchCorrectSize += threadCorrectSize;

#ifndef CLOSEOPENOMP 
#pragma omp critical 
#endif
            batchObjLoss += threadObjLoss;
        
        }  // end multi-processor

        NNet<XPU>::UpdateCumulateGrads(batchCumulatedGrads, &netsParas);
    }

    ShutdownTensorEngine<XPU>();
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cout;
    using std::endl;


    m_featExtractor.reset(new FeatureExtractor());
    m_featExtractor->getDictionaries(goldSet);

    m_transitionSystem.reset(new ActionStandardSystem());
    m_transitionSystem->makeTransition(m_featExtractor->getKnownLabels());

    m_fEmb.reset(new FeatureEmbedding(m_featExtractor->size(),
            CConfig::nFeatureNum,
            CConfig::nEmbeddingDim,
            1)); // TODO ?

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

    for (auto &gExample : gExamples) {
        for (auto &example : gExample.examples) {
            trainExamplePtrs.push_back(&(example));
        }
    }
}

State* GreedyChunker::decode(Instance *inst, NNetPara<XPU> &paras, State *lattice) {
    const static int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transitionSystem->nActNum;

    int nSentLen = inst->input.size();
    int nMaxRound = nSentLen;
    ActionStandardSystem &tranSystem = *(m_transitionSystem.get());
    FeatureExtractor &featExtractor = *(m_featExtractor.get());
    FeatureEmbedding &fEmb = *(m_fEmb.get());
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
       
        std::vector<std::vector<int>> featureVectors;
        featureVectors.resize(1);
        generateInputBatch(currentState, inst, featureVectors);
        fEmb.returnInput(featureVectors, input);

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

void GreedyChunker::generateInputBatch(State *state, Instance *inst, std::vector<std::vector<int>> &featvecs) {
    for (int i = 0; i < featvecs.size(); i++) {
        featvecs[i].resize(CConfig::nFeatureNum);
        m_featExtractor->extractFeature(*(state + i), *inst, featvecs[i]);
    }
}
