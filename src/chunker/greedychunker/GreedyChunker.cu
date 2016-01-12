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

std::pair<GreedyChunker::ChunkedResultType, GreedyChunker::ChunkedResultType> GreedyChunker::chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<XPU> &modelParas) {
    static int chunkRound = 1;
    static auto longestInst = *std::max_element(devInstances.begin(), devInstances.end(), [](Instance &inst1, Instance &inst2) { return inst1.size() < inst2.size();} );
    std::vector<State *> lattices(CConfig::nThread);
    for (int i = 0; i < lattices.size(); i++) {
        lattices[i] = new State[longestInst.size() + 1];
    }

    clock_t start, end;
    start = clock();
    ChunkedDataSet predDevSet;
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        Instance &currentInstance = devInstances[inst];
        predDevSet.push_back(LabeledSequence(currentInstance.input));
    }

#pragma omp parallel num_threads(CConfig::nThread)
    {
        int threadIndex =  omp_get_thread_num();

        for (unsigned inst = threadIndex; inst < devInstances.size(); inst += CConfig::nThread) {
            Instance &currentInstance = devInstances[inst];

            State* predState = decode(&currentInstance, modelParas, lattices[threadIndex]);

            LabeledSequence &predSent = predDevSet[inst];

            m_transSystemPtr->generateOutput(*predState, predSent);
        }
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr << "[" << chunkRound << "] totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl; chunkRound++;

    for (int i = 0; i< lattices.size(); i++) {
        delete []lattices[i];
    }

    ChunkedResultType FB1 = Evalb::eval(predDevSet, goldDevSet);
    ChunkedResultType NPFB1 = Evalb::eval(predDevSet, goldDevSet, true);

    return std::make_pair(FB1, NPFB1);
}

void GreedyChunker::printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, Model<XPU> &modelParas, double batchObjLoss, double posClassificationRate, ChunkedResultType &bestDevFB1, ChunkedResultType &bestDevNPFB1) {
    auto res = chunk(devSet, devGoldSet, modelParas);

    ChunkedResultType &currentFB1 = std::get<0>(res);
    ChunkedResultType &currentNPFB1 = std::get<1>(res);
    if (std::get<2>(currentFB1) > std::get<2>(bestDevFB1)) {
        bestDevFB1 = currentFB1;
    }
    if (std::get<2>(currentNPFB1) > std::get<2>(bestDevNPFB1)) {
        bestDevNPFB1 = currentNPFB1;
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

void display1Tensor( Tensor<XPU, 1, real_t> & tensor ){
    for(int i = 0; i < tensor.size(0); i++)
        std::cerr<<tensor[i]<<" ";
    std::cerr<<std::endl;
}

void display2Tensor( Tensor<XPU, 2, double> tensor ){
    std::cerr<<"size 0 :" << tensor.size(0)<<" size 1: "<<tensor.size(1)<<std::endl;
    for(int i = 0; i < tensor.size(0); i++){
       for(int j = 0; j < tensor.size(1); j++)
           std::cerr<<tensor[i][j]<<" ";
       std::cerr<<std::endl;
    }
}

void GreedyChunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
    std::cerr << "[trainingSet involved initing]Initing DictManager &  FeatureManager & ActionStandardSystem & generateTrainingExamples..." << std::endl;
    initTrain(trainGoldSet, trainSet);

    std::cerr << "[devSet involved initing]Initing generateInstanceSetCache for devSet..." << std::endl;
    initDev(devSet);

    const static int num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transSystemPtr->getActNumber();
    const static int batchSize = std::min(CConfig::nGreedyBatchSize, static_cast<int>(trainExamplePtrs.size()));
    // const static int batchSize = static_cast<int>(trainExamplePtrs.size());

    srand(0);

    Stream<XPU> *sstream = NewStream<XPU>();
    InitTensorEngine<XPU>();

    auto featureTypes = m_featManagerPtr->getFeatureTypes();

    std::cerr << "[begin]featureTypes:" << std::endl;
    for (auto &ft : featureTypes) {
        std::cerr << "  " << ft.typeName << ":" << std::endl;
        std::cerr << "    dictSize = " << ft.dictSize << std::endl;
        std::cerr << "    featSize = " << ft.featSize << std::endl;
        std::cerr << "    embsSize = " << ft.featEmbSize << std::endl;
    }
    std::cerr << "[end]" << std::endl;

    Model<XPU> modelParas(num_in, num_hidden, num_out, featureTypes, sstream, true);
    // m_featEmbManagerPtr->readPretrainedEmbeddings(modelParas);

    Model<XPU> adaGradSquares(num_in, num_hidden, num_out, featureTypes, sstream, false);

    ChunkedResultType bestDevFB1 = std::make_tuple(0.0, 0.0, -1.0);
    ChunkedResultType bestDevNPFB1 = std::make_tuple(0.0, 0.0, -1.0);

    int batchCorrectSize = 0;
    double batchObjLoss = 0.0;

    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double posClassificationRate = 100 * static_cast<double>(batchCorrectSize) / batchSize;

            // std::cerr << "2-norm: " << modelParas.norm2() << std::endl;
            // std::cerr << "2-embedding-norm: " << modelParas.embeddings_norm2() << std::endl;
            printEvaluationInfor(devSet, devGoldSet, modelParas, batchObjLoss + 0.5 * CConfig::fRegularizationRate * modelParas.norm2(), posClassificationRate, bestDevFB1, bestDevNPFB1);
        }
        batchCorrectSize = 0;
        batchObjLoss = 0.0;

        clock_t start, end;
        start = clock();

        // random shuffle the training instances in the container,
        // and assign them for each threads
        std::vector<ExamplePtrs> multiThread_miniBatch_data;
        generateMultiThreadsMiniBatchData(multiThread_miniBatch_data);

        Model<XPU> batchCumulatedGrads(num_in, num_hidden, num_out, featureTypes, sstream, false);
        
// #pragma omp parallel num_threads(CConfig::nThread)
        {
            int threadIndex = omp_get_thread_num();
            auto currentThreadData = multiThread_miniBatch_data[threadIndex];

            int threadCorrectSize = 0;
            double threadObjLoss = 0.0;

            Model<XPU> cumulatedGrads(num_in, num_hidden, num_out, featureTypes, sstream, false);
            std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(CConfig::nGPUBatchSize, num_in, num_hidden, num_out, &modelParas));

            std::vector<FeatureVector> featureVectors(CConfig::nGPUBatchSize);
            TensorContainer<cpu, 2, real_t> input(Shape2(CConfig::nGPUBatchSize, num_in));

            std::vector<std::vector<int>> validActsVec(CConfig::nGPUBatchSize);
            TensorContainer<cpu, 2, real_t> pred(Shape2(CConfig::nGPUBatchSize, num_out));

            for (unsigned inst = 0; inst < currentThreadData.size(); inst += static_cast<unsigned>(CConfig::nGPUBatchSize)) {
                input = 0.0;
                pred  = 0.0;
                for (unsigned insti = 0; insti < static_cast<unsigned >(CConfig::nGPUBatchSize); insti++) {
                    Example *e = currentThreadData[inst + insti];

                    featureVectors[insti] = e->features;
                    validActsVec[insti] = e->labels;
                }
                m_featEmbManagerPtr->returnInput(featureVectors, modelParas.featEmbs, input);

                nnet->Forward(input, pred, false);

                for (unsigned insti = 0; insti < static_cast<unsigned>(CConfig::nGPUBatchSize); insti++) {
                    int optAct = -1;
                    int goldAct = -1;

                    std::vector<int> &validActs = validActsVec[insti];
                    for (int i = 0; i < validActs.size(); i++) {
                        if (validActs[i] >= 0) {
                            if (optAct == -1 || pred[insti][i] > pred[insti][optAct]){
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

                    real_t maxScore = pred[insti][optAct];
                    real_t goldScore = pred[insti][goldAct];

                    real_t sum = 0.0;
                    for (int i = 0; i < validActs.size(); i++) {
                        if (validActs[i] >= 0) {
                            pred[insti][i] = std::exp(pred[insti][i] - maxScore);
                            sum += pred[insti][i];
                        }
                    }

                    threadObjLoss += (std::log(sum) - (goldScore - maxScore)) / batchSize;

                    for (int i = 0; i < validActs.size(); i++) {
                        if (validActs[i] >= 0) {
                            pred[insti][i] = pred[insti][i] / sum;
                        } else {
                            pred[insti][i] = 0.0;
                        }
                    }
                    pred[insti][goldAct] -= 1.0;
                }

                pred /= static_cast<real_t>(batchSize);

                nnet->Backprop(pred);
                nnet->SubsideGradsTo(&cumulatedGrads, featureVectors);
            }

// #pragma omp barrier
// #pragma omp critical 
            batchCumulatedGrads.mergeModel(&cumulatedGrads);

// #pragma omp critical 
            batchCorrectSize += threadCorrectSize;

// #pragma omp critical 
            batchObjLoss += threadObjLoss;
        
        }  // end multi-processor

        modelParas.update(&batchCumulatedGrads, &adaGradSquares);

        end = clock();
        if (iter % CConfig::nEvaluatePerIters == 0) 
        {
            double time_used = (double)(end - start) / CLOCKS_PER_SEC;
            std::cerr << "[" << iter << "] totally train " << batchSize << " examples, time: " << time_used << " average: " << batchSize / time_used << " examples/second!" << std::endl; 
        }
    }

    ShutdownTensorEngine<XPU>();
}

void GreedyChunker::initDev(InstanceSet &devSet) {
    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), devSet);
}

void GreedyChunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
    using std::cerr;
    using std::endl;

    m_dictManagerPtr.reset(new DictManager());
    m_dictManagerPtr->init(goldSet);

    m_featManagerPtr.reset(new FeatureManager());
    m_featManagerPtr->init(goldSet, m_dictManagerPtr);

    m_featEmbManagerPtr.reset(new FeatureEmbeddingManager(
                m_featManagerPtr->getFeatureTypes(),
                m_featManagerPtr->getDictManagerPtrs(),
                static_cast<real_t>(CConfig::fInitRange)));

    std::cerr << "  total input embedding dim: " << m_featEmbManagerPtr->getTotalFeatEmbSize() << std::endl;
    m_transSystemPtr.reset(new ActionStandardSystem());
    m_transSystemPtr->init(goldSet);

    Instance::generateInstanceSetCache(*(m_dictManagerPtr.get()), trainSet);

    GlobalExample::generateTrainingExamples(*(m_transSystemPtr.get()), *(m_featManagerPtr.get()), trainSet, goldSet, gExamples);

    for (auto &gExample : gExamples) {
        for (auto &example : gExample.examples) {
            trainExamplePtrs.push_back(&(example));
        }
    }

    std::cerr << "  Greedy train set size: " << trainExamplePtrs.size() << std::endl;
}

State* GreedyChunker::decode(Instance *inst, Model<XPU> &modelParas, State *lattice) {
    const static int num_in = m_featEmbManagerPtr->getTotalFeatEmbSize();
    const static int num_hidden = CConfig::nHiddenSize;
    const static int num_out = m_transSystemPtr->getActNumber();

    int nSentLen = inst->input.size();
    int nMaxRound = nSentLen;
    ActionStandardSystem &tranSystem = *(m_transSystemPtr.get());
    std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, num_in, num_hidden, num_out, &modelParas));

    State *retval = nullptr;
    for (int i = 0; i < nMaxRound + 1; ++i) {
        lattice[i].m_nLen = nSentLen;
    }

    lattice[0].clear();

    TensorContainer<cpu, 2, real_t> input(Shape2(1, num_in));
    TensorContainer<cpu, 2, real_t> pred(Shape2(1, num_out));
       
    for (int nRound = 1; nRound <= nMaxRound; nRound++){
        input = 0.0;
        pred = 0.0;

        State *currentState = lattice + nRound - 1;
        State *target = lattice + nRound;

        std::vector<FeatureVector> featureVectors(1);
        // featureVectors[0].clear();
        generateInputBatch(currentState, inst, featureVectors);
        m_featEmbManagerPtr->returnInput(featureVectors, modelParas.featEmbs, input);

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

    return retval;
}

void GreedyChunker::generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs) {
    for (int i = 0; i < featvecs.size(); i++) {
        m_featManagerPtr->extractFeature(*(state + i), *inst, featvecs[i]);
    }
}
