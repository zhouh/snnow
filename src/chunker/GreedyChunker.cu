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
// #define DEBUG2
// #define DEBUG3
// #define DEBUG4
// #define DEBUG5
// #define DEBUG6
// #define DEBUG7
// #define CONSTROUNDDEBUG
// #define ADDREGURLOSS
#define DEBUG8
// #define DEBUG9
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

        ChunkedSentence &predSent = predDevSet[inst];

#ifdef DEBUG9
        std::cout << "Before chunked: " << std::endl;
        std::cout << predSent << std::endl;
#endif
        m_transitionSystem->generateOutput(*predState, predSent);

#ifdef DEBUG9
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

void GreedyChunker::printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, NNetPara<XPU> &netsPara, double batchObjLoss, double posClassificationRate, double &bestDevFB1) {
    static int iter = 0;
    iter++;
#ifdef DEBUG9
    std::cout << "iter = " << iter << std::endl;
#endif
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
    paraLoss *= 0.5 * CConfig::fRegularizationRate;

    loss += paraLoss;
#endif

    std::cout << "current iteration FB1-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << currentFB1 << "\tbest FB1-score: " << bestDevFB1 << std::endl;
    std::cout << "current objective fun-score: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << loss << "\tclassfication rate: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << posClassificationRate << std::endl;
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

void GreedyChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cout << "Initing FeatureExtractor & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cout << "Excuting generateInstanceSetCache & readPretrainEmbeddings..." << std::endl;
    m_featExtractor->generateInstanceSetCache(devSet);

    m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);

    const static int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transitionSystem->nActNum;
    const static int batchSize = std::min(CConfig::nBatchSize, static_cast<int>(trainExamplePtrs.size()));

    // omp_set_num_threads(CConfig::nThread);

    srand(0);

    NNetPara<XPU> netsParas(1, num_in, num_hidden, num_out);

    double bestDevFB1 = -1.0;

    int batchCorrectSize = 0;
    double batchObjLoss = 0.0;
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        std::cout << "\nPress any key:" << std::endl;
        char tch;
        std::cin >> tch;
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double posClassificationRate = 100 * static_cast<double>(batchCorrectSize) / batchSize;

            printEvaluationInfor(devSet, devGoldSet, netsParas, batchObjLoss, posClassificationRate, bestDevFB1);
        }
        batchCorrectSize = 0;
        batchObjLoss = 0.0;

        // random shuffle the training instances in the container,
        // and assign them for each threads
        std::vector<ExamplePtrs> multiThread_miniBatch_data;
        // std::cout << "initialize multiThread_miniBatch_data" << std::endl;

        // prepare mini-batch data for each threads
        // std::random_shuffle(trainExamplePtrs.begin(), trainExamplePtrs.end());
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

// #pragma omp parallel
        {
            // auto currentThreadData = multiThread_miniBatch_data[omp_get_thread_num()];
            auto currentThreadData = multiThread_miniBatch_data[0];
            UpdateGrads<XPU> cumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);
            int threadCorrectSize = 0;
            double threadObjLoss = 0.0;

            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                Example *e = currentThreadData[inst];
                std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &netsParas));

                InitTensorEngine<XPU>();
                TensorContainer<cpu, 2, real_t> input;
                input.Resize(Shape2(1, num_in));

                TensorContainer<cpu, 2, real_t> pred;
                pred.Resize(Shape2(1, num_out));

                std::vector<std::vector<int>> featureVectors;
                featureVectors.push_back(e->features);
                m_fEmb->returnInput(featureVectors, input);
#ifdef DEBUG8
                std::cout << "Feature input index: ";
                for (int i = 0; i < CConfig::nFeatureNum; i++) {
                    std::cout << e->features[i] << " ";
                }
                std::cout << std::endl;
#endif

                nnet->Forward(input, pred, false);

                std::vector<int> validActs(e->labels);
#ifdef DEBUG8
                std::cout << "[valid acts sequence]: ";
                for (int ai = 0; ai < validActs.size(); ai++){
                    std::cout << validActs[ai] << " ";
                }
                std::cout << std::endl;
                // std::cout <<"[nn scores]: ";
                // for (int ai = 0; ai < validActs.size(); ai++) {
                //     std::cout << pred[0][ai] << " ";
                // }
                // std::cout << std::endl;
#endif 
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
#ifdef DEBUG8
                std::cout << "maxAct = " << optAct << "\tgoldAct = " << goldAct << std::endl;
                std::cout << "maxScore = " << maxScore << "\tgoldScore = " << goldScore << std::endl;
#endif
                real_t sum = 0.0;
                for (int i = 0; i < validActs.size(); i++) {
                    if (validActs[i] >= 0) {
                        pred[0][i] = std::exp(pred[0][i] - maxScore);
                        sum += pred[0][i];
                    }
                }

#ifdef DEBUG8
                // std::cout << "sum = " << sum << "\tlog(sum) = " << std::log(sum) << "\tlog(gold) = " << std::log(std::exp(goldScore - maxScore)) << std::endl;
#endif

                threadObjLoss += (std::log(sum) - (goldScore - maxScore)) / batchSize;
#ifdef DEBUG8
                // std::cout <<"[divided by exp(maxScore)]: ";
                // for (int ai = 0; ai < validActs.size(); ai++) {
                //     std::cout << pred[0][ai] << " ";
                // }
                // std::cout << std::endl;
                // std::cout << "threadObjLoss: " << threadObjLoss << std::endl;
#endif
                for (int i = 0; i < validActs.size(); i++) {
                    if (validActs[i] >= 0) {
                        pred[0][i] = pred[0][i] / sum;
                    } else {
                        pred[0][i] = 0.0;
                    }
                }
                pred[0][goldAct] -= 1.0;
#ifdef DEBUG8
                std::cout <<"[probability]: ";
                for (int ai = 0; ai < validActs.size(); ai++) {
                    std::cout << pred[0][ai] << " ";
                }
                std::cout << std::endl;
#endif
                nnet->Backprop(pred);
                nnet->SubsideGrads(cumulatedGrads);

                ShutdownTensorEngine<XPU>();
            }

// #pragma omp barrier
// #pragma omp critical 
            {
                NNet<XPU>::UpdateCumulateGrads(cumulatedGrads, &netsParas);

                batchCorrectSize += threadCorrectSize;

                batchObjLoss += threadObjLoss;

            }
        }  // end multi-processor

//             auto longestSent = *std::max_element(currentThreadData.begin(), currentThreadData.end(), [](GlobalExample *g1, GlobalExample *g2) { return g1->instance.size() < g2->instance.size();} );
// 
//             State *lattice = new State[longestSent->instance.size() + 1];
// 
//             // for every instance in this mini-batch
//             for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
//                 // fetch a to-be-trained instance
//                 GlobalExample *ge = currentThreadData[inst];
//                 int nMaxRound = static_cast<int>(ge->instance.size());
// 
//                 lattice[0].clear();
//                 for (int i = 0; i <= nMaxRound; i++) {
//                     lattice[i].m_nLen = nMaxRound;
//                 }
// 
//                 InitTensorEngine<XPU>();
//                 for (int nRound = 1; nRound <= nMaxRound; nRound++){
//                     std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &netsParas));
// #ifdef CONSTROUNDDEBUG
//                     const int roundConstant = 16;
//                     if (nRound != roundConstant) 
//                         continue;
// #endif
//                     State *currentState = lattice + nRound - 1;
//                     State *target = lattice + nRound;
//                     int goldAct = ge->goldActs[nRound - 1];
// 
//                     TensorContainer<cpu, 2, real_t> input;
//                     input.Resize(Shape2(1, num_in));
// 
//                     TensorContainer<cpu, 2, real_t> pred;
//                     pred.Resize(Shape2(1, num_out));
// 
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant)
// #endif
//                     std::cout << "iter = " << iter << ", inst = " << inst << ", nRound = " << nRound << std::endl; 
// #endif
//                     std::vector<std::vector<int>> featureVectors(1);
//                     generateInputBatch(currentState, &(ge->instance), featureVectors);
//                     m_fEmb->returnInput(featureVectors, input);
// 
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant){
// #endif
//                     std::cout << "Feature input value: ";
//                     for (int i = 0; i < CConfig::nFeatureNum; i++) {
//                         std::cout << input[0][i] << " ";
//                     }
//                     std::cout << std::endl;
// #ifdef CONSTROUNDDEBUG
//                     }
// #endif
// #endif
//                     nnet->Forward(input, pred, false);
// 
//                     std::vector<int> validActs;
//                     m_transitionSystem->generateValidActs(*currentState, validActs);
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant){
// #endif
//                     std::cout << "[valid acts sequence]: ";
//                     for (int ai = 0; ai < validActs.size(); ai++){
//                         std::cout << validActs[ai] << " ";
//                     }
//                     std::cout << std::endl;
//                     std::cout <<"[nn scores]: ";
//                     for (int ai = 0; ai < validActs.size(); ai++) {
//                         std::cout << pred[0][ai] << " ";
//                     }
//                     std::cout << std::endl;
// #ifdef CONSTROUNDDEBUG
//                     }
// #endif
// #endif
// 
//                     int optAct = -1;
//                     for (int i = 0; i < validActs.size(); i++) {
//                         if (i == goldAct || validActs[i] >= 0) {
//                             if (optAct == -1 || pred[0][i] > pred[0][optAct]){
//                                 optAct = i;
//                             }
//                         }
//                     }
//                     if (optAct == goldAct) {
//                         threadCorrectSize += 1;
//                     }
// 
//                     real_t maxScore = pred[0][optAct];
//                     real_t goldScore = pred[0][goldAct];
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant) {
// #endif
//                     std::cout << "maxAct = " << optAct << "\tgoldAct = " << goldAct << std::endl;
//                     std::cout << "maxScore = " << maxScore << "\tgoldScore = " << goldScore << std::endl;
// #ifdef CONSTROUNDDEBUG
//                     }
// #endif
// #endif
//                     real_t sum = 0.0;
//                     for (int i = 0; i < validActs.size(); i++) {
//                         if (i == goldAct || validActs[i] >= 0) {
//                             pred[0][i] = std::exp(pred[0][i] - maxScore);
//                             sum += pred[0][i];
//                         }
//                     }
// #ifdef DEBUG2
//                     std::cout << "log(sum) = " << std::log(sum) << "\tlog(gold) = " << std::log(std::exp(goldScore - maxScore)) << std::endl;
// #endif
// 
//                     threadObjLoss += (std::log(sum) - (goldScore - maxScore)) / batchSize;
// 
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant) {
// #endif
//                     std::cout << "sum = " << sum << std::endl;
//                     std::cout <<"[divided by exp(maxScore)]: ";
//                     for (int ai = 0; ai < validActs.size(); ai++) {
//                         std::cout << pred[0][ai] << " ";
//                     }
//                     std::cout << std::endl;
//                     std::cout << "threadObjLoss: " << threadObjLoss << std::endl;
// #ifdef CONSTROUNDDEBUG
//                     }
// #endif
// #endif
// 
//                     for (int i = 0; i < validActs.size(); i++) {
//                         if (i == goldAct || validActs[i] >= 0) {
//                             pred[0][i] = pred[0][i] / sum;
//                         } else {
//                             pred[0][i] = 0.0;
//                         }
//                     }
//                     pred[0][goldAct] -= 1.0;
// 
// #ifdef DEBUG7
// #ifdef CONSTROUNDDEBUG
//                     if (nRound == roundConstant) {
// #endif
//                     std::cout <<"[probability]: ";
//                     for (int ai = 0; ai < validActs.size(); ai++) {
//                         std::cout << pred[0][ai] << " ";
//                     }
//                     std::cout << std::endl;
// #ifdef CONSTROUNDDEBUG
//                     }
// #endif
// #endif
// 
//                     for (int i = 0; i < validActs.size(); i++) {
//                         pred[0][i] /= batchSize;
//                     }
// 
//                     nnet->Backprop(pred);
//                     nnet->SubsideGrads(cumulatedGrads);
// 
//                     CScoredTransition trans(currentState, goldAct, currentState->score + goldScore);
//                     *target = *currentState;
//                     m_transitionSystem->move(*currentState, *target, trans);
//                 }
//                 ShutdownTensorEngine<XPU>();
//             } // end for instance traverse
// 
//             delete []lattice;
// 
// #pragma omp barrier
// #pragma omp critical 
//             {
//                 NNet<XPU>::UpdateCumulateGrads(cumulatedGrads, &netsParas);
// 
//                 batchCorrectSize += threadCorrectSize;
// 
//                 batchObjLoss += threadObjLoss;
// 
//             }
//         }  // end multi-processor
    }
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cout;
    using std::endl;


    m_featExtractor.reset(new FeatureExtractor());
    m_featExtractor->getDictionaries(goldSet);

    m_transitionSystem.reset(new ActionStandardSystem());
    m_transitionSystem->makeTransition(m_featExtractor->getKnownLabels());

#ifdef DEBUGX
    m_transitionSystem->displayLabel2ActionIdx();
#endif

    m_fEmb.reset(new FeatureEmbedding(m_featExtractor->size(),
            CConfig::nFeatureNum,
            CConfig::nEmbeddingDim,
            1)); // TODO ?

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

#ifdef DEBUGX
    std::cout << "train set size: " << trainSet.size() << std::endl;
    std::cout << "dev gold set size: " << goldSet.size() << std::endl;
    std::cout << "global examples size: " << gExamples.size() << std::endl;
#endif
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

#ifdef DEBUG5
    std::cout << "[deco action sequances]: ";
#endif
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
#ifdef DEBUG6
        for (unsigned ii = 0; ii < validActs.size(); ++ii) {
            std::cout << "validActs[" << ii << "]=" << validActs[ii] << " "; 
        }
        std::cout << std::endl;
        for (unsigned ii = 0; ii < pred.shape_[1]; ii++) {
            std::cout << "pred[" << ii << "]=" << pred[0][ii] << " ";
        }
        std::cout << std::endl;
#endif
        for (unsigned actID = 0; actID < validActs.size(); ++actID) {
            if (validActs[actID] == -1) {
                continue;
            }

            if (actID == 0 || pred[0][actID] > maxScore) {
                maxScore = pred[0][actID];
                maxActID = actID;
            }
        }

#ifdef DEBUG5
        maxActID = example.goldActs[nRound - 1];
        std::cout << maxActID << " ";
#endif
        CScoredTransition trans(currentState, maxActID, currentState->score + maxScore);
        *target = *currentState;
        tranSystem.move(*currentState, *target, trans);
        retval = target;
#ifdef DEBUG5
        std::cout << "[" << target->last_action << "](" << trans.action << ") ";
#endif
#ifdef DEBUGX
        std::cout << "nRound = " << nRound << std::endl;
#endif
    }

#ifdef DEBUG5
    std::cout << std::endl;
#endif

    ShutdownTensorEngine<XPU>();

    return retval;
}

void GreedyChunker::generateInputBatch(State *state, Instance *inst, std::vector<std::vector<int>> &featvecs) {
        for (int i = 0; i < featvecs.size(); i++) {
            featvecs[i].resize(CConfig::nFeatureNum);
            m_featExtractor->extractFeature(*(state + i), *inst, featvecs[i]);
        }
    }
