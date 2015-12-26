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

#define XPU gpu

#include "NNet.h"
#include "Config.h"
#include "FeatureVector.h"
#include "FeatureManager.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"

class GreedyChunker {
public:
    typedef std::vector<Example *> ExamplePtrs;
private:
    LabelManager labelManager;
    std::shared_ptr<ActionStandardSystem> m_transitionSystem;
    std::shared_ptr<FeatureManager> m_featManager;

    bool m_bTrain;

    GlobalExamples gExamples;

    ExamplePtrs trainExamplePtrs;
public:
    GreedyChunker();
    GreedyChunker(bool isTrain);

    ~GreedyChunker();

    void train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet);

    double chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsPara);

private:
    void initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet);

    State *decode(Instance *inst, NNetPara<XPU> &paras, State *lattice);

    /* generate the feature vector in all the beam states,
     * and return the input layer of neural network in batch.
    */
    void generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs); 

    void printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, NNetPara<XPU> &netsPara, double batchObjLoss, double posClassificationRate, double &bestDevFB1);

    void generateMultiThreadsMiniBatchData(std::vector<ExamplePtrs> &multiThread_miniBatch_data);
};

#endif
