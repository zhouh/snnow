/*************************************************************************
	> File Name: Chunker.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 02:39:07 PM CST
 ************************************************************************/
#ifndef _CHUNKER_CHUNKER_H_
#define _CHUNKER_CHUNKER_H_ 
#include <iostream>
#include <memory>


#include "Config.h"
#include "FeatureExtractor.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "FeatureEmbedding.h"
#include "Instance.h"

#include "NNet.h"

#define XPU gpu

class Chunker{
    std::shared_ptr<FeatureExtractor> m_featExtractor;
    std::shared_ptr<ActionStandardSystem> m_transitionSystem;
    std::shared_ptr<FeatureEmbedding> m_fEmb;

    int m_nBeamSize;
    bool m_bTrain;
    bool m_bEarlyUpdate;

    GlobalExamples gExamples;
public:
    Chunker();
    Chunker(bool isTrain);

    ~Chunker();

    void train(ChunkedDataSet &goldSet, InstanceSet &trainSet, InstanceSet &devSet);

    double parse(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsPara);

private:
    void initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet);
};

#endif
