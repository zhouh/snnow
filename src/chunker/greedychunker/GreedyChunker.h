/*************************************************************************
	> File Name: GreedyChunker.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 07 Dec 2015 08:50:16 PM CST
 ************************************************************************/
#ifndef _CHUNKER_GREEDYCHUNKER_GREEDYCHUNKER_H_
#define _CHUNKER_GREEDYCHUNKER_GREEDYCHUNKER_H_

#include <iostream>
#include <memory>

#include "Config.h"
#include "chunker.h"

#include "Model.h"
#include "NNet.h"

#include "LabeledSequence.h"
#include "Instance.h"
#include "Example.h"
#include "FeatureVector.h"
#include "DictManager.h"
#include "FeatureManager.h"
#include "FeatureEmbeddingManager.h"
#include "ActionStandardSystem.h"

class GreedyChunker {
public:
    typedef std::vector<Example *> ExamplePtrs;
private:
    LabelManager labelManager;
    std::shared_ptr<ActionStandardSystem> m_transSystemPtr;
    std::shared_ptr<DictManager> m_dictManagerPtr;
    std::shared_ptr<FeatureEmbeddingManager> m_featEmbManagerPtr;
    std::shared_ptr<FeatureManager> m_featManagerPtr;

    bool m_bTrain;

    GlobalExamples gExamples;

    ExamplePtrs trainExamplePtrs;
public:
    GreedyChunker();
    GreedyChunker(bool isTrain);

    ~GreedyChunker();

    void train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet, InstanceSet &devSet);

    double chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, Model<XPU> &modelParas);

private:
    void initDev(InstanceSet &devSet);

    void initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet);

    State *decode(Instance *inst, Model<XPU> &modelParas, State *lattice);

    /* generate the feature vector in all the beam states,
     * and return the input layer of neural network in batch.
    */
    void generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs); 

    void printEvaluationInfor(InstanceSet &devSet, ChunkedDataSet &devGoldSet, Model<XPU> &modelParas, double batchObjLoss, double posClassificationRate, double &bestDevFB1);

    void generateMultiThreadsMiniBatchData(std::vector<ExamplePtrs> &multiThread_miniBatch_data);

    GreedyChunker(const GreedyChunker &chunker) = delete;
    GreedyChunker& operator= (const GreedyChunker &chunker) = delete;
};

#endif
