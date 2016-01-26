/*************************************************************************
	> File Name: BeamChunkerThread.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 22 Jan 2016 06:43:17 PM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMCHUNKER_BEAMTRAINTHREAD_H_
#define _CHUNKER_BEAMCHUNKER_BEAMTRAINTHREAD_H_

#include <vector>
#include <memory>

#include "chunker.h"
#include "FeatureType.h"
#include "Example.h"
#include "Model.h"
#include "TNNets.h"
#include "NNet.h"
#include "State.h"
#include "Instance.h"
#include "LabeledSequence.h"

#include "mshadow/tensor.h"

#include "ActionStandardSystem.h"
#include "FeatureManager.h"
#include "FeatureEmbeddingManager.h"

class BeamChunkerThread {
public:
    std::shared_ptr<ActionStandardSystem> m_transSystemPtr;
    std::shared_ptr<FeatureManager> m_featManagerPtr;
    std::shared_ptr<FeatureEmbeddingManager> m_featEmbManagerPtr;

    int m_nDeviceID;
    int m_nNumIn;
    int m_nNumHidden;
    int m_nNumOut;
    int m_nBeamSize;
    int m_nMiniBatchSize;
    Stream<gpu> *stream;
    std::shared_ptr<Model<gpu>> modelPtr;

    std::vector<NNet<gpu> *> netPtrs;
    State* statePtr;
    State** stateIndexPtr;

    BeamChunkerThread(
            const int threadId, 
            const int beamSize, 
            const int miniBatchSize,
            Model<cpu> &paraModel, 
            std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
            std::shared_ptr<FeatureManager> featureMangerPtr,
            std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
            int longestLen);

    ~BeamChunkerThread();

    void train(Model<cpu> &paraModel, std::vector<GlobalExample *> &gExamplePtrs, Model<cpu> &cumulatedGrads, double &threadLoss);

    void chunk(Model<cpu> &paraModel, InstanceSet &devInstances, std::vector<int> &threadDevInstanceIndexes, ChunkedDataSet &labeledSents);

};

#endif
