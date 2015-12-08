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
 #define DEBUG1
#endif

GreedyChunker::GreedyChunker() {

}

GreedyChunker::GreedyChunker(bool isTrain) {
    m_bTrain = isTrain;
}

GreedyChunker::~GreedyChunker() {

}

double GreedyChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsParas) {

    auto longestInst = *std::max_element(devInstances.begin(), devInstances.end(), [](Instance &inst1, Instance &inst2) { return inst1.size() < inst2.size();} );
#ifdef DEBUGX
    std::cout << "longest instance's size: " << longestInst.size() << std::endl;
#endif
    State *lattice = new State[longestInst.size() + 1];

    clock_t start, end;
    start = clock();
    ChunkedDataSet predDevSet;
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        Instance &currentInstance = devInstances[inst];
        predDevSet.push_back(ChunkedSentence(currentInstance.input));

        State* predState = decode(&currentInstance, netsParas, lattice);

#ifdef DEBUGX
        std::cout << "Current instance's size: " << devInstances[inst].size() << std::endl;
        State *ptr = predState;
        int i = 1;
        while (ptr != nullptr && ptr->last_action != -1) {
            i++;
            ptr = ptr->previous_;
        }
        std::cout << "i = " << i << std::endl;
#endif 
        ChunkedSentence &predSent = predDevSet[inst];

#ifdef DEBUGX
        std::cout << "Before chunked: " << std::endl;
        std::cout << predSent << std::endl;
#endif
        m_transitionSystem->generateOutput(*predState, predSent);

#ifdef DEBUGX
        std::cout << "After chunked: " << std::endl;
        std::cout << predSent << std::endl;
#endif
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

    delete []lattice;

#ifdef DEBUGX
    std::cout << "pred dev set's size: " << predDevSet.size() << std::endl;
    std::cout << "gold dev set's size: " << goldDevSet.size() << std::endl;
#endif
    auto res = Evalb::eval(predDevSet, goldDevSet);

    return std::get<2>(res);
}

void GreedyChunker::train(ChunkedDataSet &goldSet, InstanceSet &trainSet, InstanceSet &devSet) {
    initTrain(goldSet, trainSet);

    m_featExtractor->generateInstanceSetCache(devSet);
#ifdef DEBUGX
    std::cout << "After generateInstanceSetCache!" << std::endl;
#endif
    m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);
#ifdef DEBUGX
    std::cout << "After readPretrainEmbeddings!" << std::endl;
#endif

    const static int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transitionSystem->nActNum;

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    NNetPara<XPU> netsParas(1, num_in, num_hidden, num_out);
    double bestDevFB1 = std::numeric_limits<double>::min();
#ifdef DEBGU1
    std::cout << "Before chunk training..." << std::endl;
#endif

    for (int iter = 0; iter < CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double currentFB1 = chunk(devSet, goldSet, netsParas);
            if (currentFB1 > bestDevFB1) {
                bestDevFB1 = currentFB1;
            }
            std::cout << "current iteration FB1-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << currentFB1 << "\t best FB1-score: " << bestDevFB1 << std::endl;
        }

        // random shuffle the training instances in the container,
        // and assign them for each thread
        std::vector<std::vector<GlobalExample *>> multiThread_miniBatch_data;

        // prepare mini-batch data for each threads
        std::random_shuffle(gExamples.begin(), gExamples.end());
        int threadExampleNum = std::min(CConfig::nBatchSize, static_cast<int>(gExamples.size())) / CConfig::nThread;
        auto sp = gExamples.begin();
        auto ep = sp + threadExampleNum;
        for (int i = 0; i < CConfig::nThread; i++) {
            std::vector<GlobalExample *> threadExamples;
            for (auto p = sp; p != ep; p++) {
                threadExamples.push_back(&(*p));
            }
#ifdef DEBUG3
            std::cout << "threadExamples' size: " << threadExamples.size() << std::endl;
            std::cout << "globalExamples' size: " << gExamples.size() << std::endl;
#endif
            sp = ep;
            ep += threadExampleNum;
            multiThread_miniBatch_data.push_back(threadExamples);
        }

#pragma omp parallel
        {
            std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &netsParas));
            auto currentThreadData = multiThread_miniBatch_data[omp_get_thread_num()];
            UpdateGrads<XPU> cumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);

            auto longestSent = *std::max_element(currentThreadData.begin(), currentThreadData.end(), [](GlobalExample *g1, GlobalExample *g2) { return g1->instance.size() < g2->instance.size();} );
            int currentBatchSize = 0;
            for (GlobalExample* ge : currentThreadData) {
                currentBatchSize += ge->instance.size();
            }

            State *lattice = new State[longestSent->instance.size() + 1];

            // for evary instance in this mini-batch
            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                // fetch a to-be-trained instance
                GlobalExample *ge = currentThreadData[inst];
                int nMaxRound = ge->instance.size();

                lattice[0].clear();
                InitTensorEngine<XPU>();
                for (int nRound = 1; nRound <= nMaxRound; nRound++){
                    State *currentState = lattice + nRound - 1;
                    State *target = lattice + nRound;
                    int goldAct = ge->goldActs[nRound - 1];

                    TensorContainer<cpu, 2, real_t> input;
                    input.Resize(Shape2(1, num_in));

                    TensorContainer<cpu, 2, real_t> pred;
                    pred.Resize(Shape2(1, num_out));

                    std::vector<std::vector<int>> featureVectors(1);
                    generateInputBatch(currentState, &(ge->instance), featureVectors);
                    m_fEmb->returnInput(featureVectors, input);

                    nnet->Forward(input, pred, false);

                    std::vector<int> validActs;
                    m_transitionSystem->generateValidActs(*currentState, validActs);
                    int optAct = -1;
                    for (int i = 0; i < validActs.size(); i++) {
                        if (i == goldAct || validActs[i] >= 0) {
                            if (optAct == -1 || pred[0][i] > pred[0][optAct]){
                                optAct = i;
                            }
                        }
                    }
                    real_t maxScore = pred[0][optAct];
                    real_t goldScore = pred[0][goldAct];
                    real_t sum = 0.0;
                    for (int i = 0; i < validActs.size(); i++) {
                        if (i == goldAct || validActs[i] >= 0) {
                            pred[0][i] = exp(pred[0][i] - maxScore);
                            sum += pred[0][i];
                        }
                    }
                    for (int i = 0; i < validActs.size(); i++) {
                        if (i == goldAct || validActs[i] >= 0) {
                            pred[0][i] = pred[0][i] / sum;
                        } else {
                            pred[0][i] = 0.0;
                        }
                    }
                    pred[0][goldAct] -= 1.0;
                    for (int i = 0; i < validActs.size(); i++) {
                        pred[0][i] /= currentBatchSize;
                    }

                    nnet->Backprop(pred);
                    nnet->SubsideGrads(cumulatedGrads);

                    CScoredTransition trans;
                    trans(currentState, goldAct, currentState->score + goldScore);
                    *target = *currentState;
                    m_transitionSystem->move(*currentState, *target, trans);
                }
                ShutdownTensorEngine<XPU>();
            } // end for instance traverse

            delete []lattice;
#pragma omp barrier
#pragma omp critical
            NNet<XPU>::UpdateCumulateGrads(cumulatedGrads, &netsParas);
        }  // end multi-processor
    }
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cout;
    using std::endl;

    cout << "Training init..." << endl;
    cout << "  Training Instance num: " << trainSet.size() << endl;

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
            1)); // TODO ?

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

#ifdef DEBUG1
    std::cout << "train set size: " << trainSet.size() << std::endl;
    std::cout << "dev gold set size: " << goldSet.size() << std::endl;
    std::cout << "global examples size: " << gExamples.size() << std::endl;
#endif
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
        real_t maxScore = std::numeric_limits<real_t>::min();
        unsigned maxActID = std::numeric_limits<unsigned>::max();
        for (unsigned actID = 0; actID < validActs.size(); ++actID) {
            if (validActs[actID] == -1) {
                continue;
            }

            if (pred[0][actID] > maxScore) {
                maxScore = pred[0][actID];
                maxActID = actID;
            }
        }

        CScoredTransition trans(currentState, maxActID, currentState->score + maxScore);
        *target = *currentState;
        tranSystem.move(*currentState, *target, trans);
        retval = target;
#ifdef DEBUGX
        std::cout << "nRound = " << nRound << std::endl;
#endif
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
