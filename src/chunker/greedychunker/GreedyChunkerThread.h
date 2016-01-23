/*************************************************************************
	> File Name: GreedyChunkerThread.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 23 Jan 2016 01:00:27 PM CST
 ************************************************************************/
#ifndef _CHUNKER_GREEDYCHUNKER_GREEDYCHUNKERTHREAD_
#define _CHUNKER_GREEDYCHUNKER_GREEDYCHUNKERTHREAD_

#include <vector>
#include <memory>

#include "Example.h"
#include "FeatureVector.h"
#include "State.h"
#include "Instance.h"
#include "LabeledSequence.h"

#include "mshadow/tensor.h"

#include "ActionStandardSystem.h"
#include "FeatureManager.h"
#include "FeatureEmbeddingManager.h"

#include "Model.h"
#include "NNet.h"

class GreedyChunkerThread {
private:
    std::shared_ptr<ActionStandardSystem> m_transSystemPtr;
    std::shared_ptr<FeatureManager> m_featManagerPtr;
    std::shared_ptr<FeatureEmbeddingManager> m_featEmbManagerPtr;

    int m_nThreadId;
    int m_nNumIn;
    int m_nNumHidden;
    int m_nNumOut;
    int m_nBatchSize;

    Stream<gpu> *stream;
    std::shared_ptr<Model<gpu>> modelPtr;

    State* statePtr;

public:
    GreedyChunkerThread(
            const int threadId, 
            const int batchSize, 
            Model<cpu> &paraModel, 
            std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
            std::shared_ptr<FeatureManager> featureMangerPtr,
            std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
            int longestLen);


    ~GreedyChunkerThread();

    void train(Model<cpu> &paraModel, std::vector<Example *> &examplePtrs, const int miniBatchSize, Model<cpu> &cumulatedGrads, int &threadCorrectSize, double &threadObjLoss);

    void chunk(const int threads_num, Model<cpu> &paraModel, InstanceSet &devInstances, ChunkedDataSet &labeledSents);

private:
    State* decode(Instance *inst);

    void generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs);

};

#endif
