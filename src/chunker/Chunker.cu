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
#define DEBUG4
#define DEBUG5
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

    // ChunkedDataSet predSents(devInstances.size());

    clock_t start, end;
    start = clock();
    ChunkedDataSet predictDevSet;
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        // std::cout << "inst = " << inst << std::endl;
        // char tch;
        // std::cin >> tch;
        // predSents[inst].init(devInstances[inst].input);

        ChunkedSentence predictSent(devInstances[inst].input);

        BeamDecoder decoder(&(devInstances[inst]), 
                            m_transitionSystem.get(),
                            m_featExtractor.get(),
                            m_fEmb.get(),
                            m_nBeamSize, 
                            false);

        decoder.generateChunkedSentence(tnnets, predictSent);
#ifdef DEBUG1
        std::cout << "predictSent's size: " << predictSent.m_lChunkedWords.size() << std::endl;
#endif
        //State *predState = new State();
        // m_transitionSystem->generateOutput(*predState, predictSent);

        predictDevSet.push_back(predictSent);
    }
    end = clock();

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "totally chunk " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

    auto res = Evalb::eval(predictDevSet, goldDevSet);

    return std::get<2>(res);
}

void Chunker::train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet) {
#ifdef DEBUG2
    char ch;
#endif 
    initTrain(trainGoldSet, trainSet);

    m_featExtractor->generateInstanceSetCache(devSet);

    m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);

    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transitionSystem->nActNum;

    const int beam_size = CConfig::nBeamSize;

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    NNetPara<XPU> netsParas(beam_size, num_in, num_hidden, num_out);
    for (int ii = 0; ii < netsParas.Wi2h.shape_[0]; ii++) {
        for (int jj = 0; jj < netsParas.Wi2h.shape_[1]; jj++){
            if (netsParas.Wi2h[ii][jj]) {
                std::cout << "[Chunker train [1]: NaN appears in netsParas.Wi2h" << std::endl;
                char ch;
                std::cin >> ch;
            }
        }
    }
    double bestDevFB1 = -1.0;
#ifdef DEBUG2
    std::cout << "Before chunking..." << std::endl;
#endif 
    for (int iter = 1; iter <= CConfig::nRound; iter++) {
        // Evaluate FB1 score per iteration
        std::cout << "iter = " << iter << std::endl;
        if (iter == 2) {
            std::cout << "begin to debug" << std::endl;
        }
        if (iter % CConfig::nEvaluatePerIters == 0) {
            double currentFB1 = chunk(devSet, devGoldSet, netsParas);
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

        // begin to multi thread Training
// #pragma omp parallel
        {
            auto currentThreadData = multiThread_miniBatch_data[omp_get_thread_num()];
            UpdateGrads<XPU> cumulatedGrads(netsParas.stream, num_in, num_hidden, num_out);

            // for evary instance in this mini-batch
            for (unsigned inst = 0; inst < currentThreadData.size(); inst++) {
                // fetch a to-be-trained instance
                GlobalExample *example = currentThreadData[inst];

                TNNets tnnets(m_nBeamSize, num_in, num_hidden, num_out, &netsParas);
                for (int ii = 0; ii < netsParas.Wi2h.shape_[0]; ii++) {
                    for (int jj = 0; jj < netsParas.Wi2h.shape_[1]; jj++){
                        if (netsParas.Wi2h[ii][jj]) {
                            std::cout << "[Chunker train [2]: NaN appears in netsParas.Wi2h" << std::endl;
                            char ch;
                            std::cin >> ch;
                        }
                    }
                }

                // decode and update
                // std::cout << "begin to decode!" << std::endl;
                BeamDecoder decoder(&(example->instance), 
                                    m_transitionSystem.get(),
                                    m_featExtractor.get(),
                                    m_fEmb.get(),
                                    m_nBeamSize, 
                                    true);

                std::cout << "round: " << iter << "\tinst: " << inst << std::endl;

                State * predState = decoder.decode(tnnets, example);

                std::cout << "current beamsize: " << decoder.beam.currentBeamSize << std::endl;
                // std::cout << "end decoding!" << std::endl;

                // std::vector<int> predictedActions;
                // State *ptr = predState;
                // if (ptr == nullptr) {
                //     std::cout << "predstate is nullptr!" << std::endl;
                // }
                // int i = 1;
                // while (ptr != nullptr && ptr->last_action != -1) {
                //     i++;
                //     predictedActions.push_back(ptr->last_action);
                //     ptr = ptr->previous_;
                // }
                // std::cout << "decoded path length: " << i << std::endl;
                // std::cout << "[pred action sequences]: ";
                // for (int i = 0; i < predictedActions.size(); i++) {
                //     std::cout << predictedActions[predictedActions.size() - 1 - i] << " ";
                // }
                // std::cout << std::endl;
                // char tch;
                // std::cin >> tch;

                tnnets.updateTNNetParas(cumulatedGrads, decoder.beam, decoder.bEarlyUpdate, decoder.nGoldTransitionIndex, decoder.goldScoredTran);
                for (int ii = 0; ii < cumulatedGrads.cg_Wi2h.shape_[0]; ii++) {
                    for (int jj = 0; jj < cumulatedGrads.cg_Wi2h.shape_[1]; jj++){
                        if (isnan(cumulatedGrads.cg_Wi2h[ii][jj])) {
                            std::cout << "W(input -> hidden): NaN appears!" << std::endl;
                        }
                    }
                }
                for (int ii = 0; ii < cumulatedGrads.cg_Wh2o.shape_[0]; ii++) {
                    for (int jj = 0; jj < cumulatedGrads.cg_Wh2o.shape_[1]; jj++){
                        if (isnan(cumulatedGrads.cg_Wh2o[ii][jj])) {
                            std::cout << "W(hidden -> output): NaN appears!" << std::endl;
                        }
                    }
                }
                for (int ii = 0; ii < cumulatedGrads.cg_hbias.shape_[0]; ii++) {
                    if (isnan(cumulatedGrads.cg_hbias[ii])) {
                        std::cout << "Bias: NaN appears!" << std::endl;
                    }
                }
            } // end for instance traverse

// #pragma omp barrier
// #pragma omp critical

            NNet<XPU>::UpdateCumulateGrads(cumulatedGrads, &netsParas);
        } // end multi-processor
    } // end total iteration
}
    
void Chunker::initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet) {
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
            m_nBeamSize));

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);

#ifdef DEBUG1
    std::cout << "train set size: " << trainSet.size() << std::endl;
    std::cout << "dev gold set size: " << goldSet.size() << std::endl;
    std::cout << "global examples size: " << gExamples.size() << std::endl;
#endif
}
