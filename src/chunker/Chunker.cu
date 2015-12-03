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

#include "Config.h"

#include "Beam.h"
#include "Chunker.h"
#include "TNNets.h"


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
    
double Chunker::parse(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsParas) {
    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transitionSystem->nActNum;
    const int beam_size = CConfig::nBeamSize;

    TNNets tnnets(beam_size, num_in, num_hidden, num_out, &netsParas, false);

    ChunkedDataSet predSents(devInstances.size());

    clock_t start, end;
    start = clock();
    for (unsigned inst = 0; inst < devInstances.size(); inst++) {
        predSents[inst].init(devInstances[inst].input);
    }
    end = clock();



    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "totally parse " << devInstances.size() << " sentences, time: " << time_used << " average: " << devInstances.size() / time_used << " sentences/second!" << std::endl;

    return 0.0;
}

void Chunker::train(ChunkedDataSet &goldSet, InstanceSet &trainSet, InstanceSet &devSet) {
    initTrain(goldSet, trainSet);

    m_featExtractor->generateInstanceSetCache(devSet);

    m_featExtractor->readPretrainEmbeddings(CConfig::strEmbeddingPath, *m_fEmb);

    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = m_transitionSystem->nActNum;

    const int beam_size = CConfig::nBeamSize;

    omp_set_num_threads(CConfig::nThread);

    srand(0);

    NNetPara<XPU> netsParas(beam_size, num_in, num_hidden, num_out);
    double bestDevFB1 = -1.0;

    
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
    m_transitionSystem->displayLabel2ActionIdx();

    m_fEmb.reset(new FeatureEmbedding(m_featExtractor->size(),
            CConfig::nFeatureNum,
            CConfig::nEmbeddingDim,
            m_nBeamSize));

    m_featExtractor->generateTrainingExamples(*(m_transitionSystem.get()), trainSet, goldSet, gExamples);
}
