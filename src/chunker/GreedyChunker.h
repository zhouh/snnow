/*************************************************************************
	> File Name: GreedyChunker.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 07 Dec 2015 08:50:16 PM CST
 ************************************************************************/
#ifndef _CHUNKER_GREEDYCHUNKER_H_
#define _CHUNKER_GREEDYCHUNKER_H_

#include <iostream>
#include <memory>

#define XPU cpu

#include "NNet.h"
#include "Config.h"
#include "FeatureExtractor.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "FeatureEmbedding.h"
#include "Instance.h"

class GreedyChunker {
    std::shared_ptr<FeatureExtractor> m_featExtractor;
    std::shared_ptr<ActionStandardSystem> m_transitionSystem;
    std::shared_ptr<FeatureEmbedding> m_fEmb;

    bool m_bTrain;

    GlobalExamples gExamples;
public:
    GreedyChunker();
    GreedyChunker(bool isTrain);

    ~GreedyChunker();

    void train(ChunkedDataSet &goldSet, InstanceSet &trainSet, InstanceSet &devSet);

    double chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsPara);

private:
    void initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet);

    State *decode(Instance *inst, NNetPara<XPU> &paras, State *lattice);

    /* generate the feature vector in all the beam states,
     * and return the input layer of neural network in batch.
    */
    void generateInputBatch(State *state, Instance *inst, std::vector<std::vector<int>> &featvecs); 
};

#endif
