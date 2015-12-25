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

#define XPU gpu

#include "NNet.h"
#include "Beam.h"
#include "BeamDecoder.h"
#include "Config.h"
#include "FeatureManager.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"

class Chunker{
private:
    std::shared_ptr<ActionStandardSystem> m_transitionSystem;
    std::shared_ptr<FeatureManager> m_featManager;

    int m_nBeamSize;
    bool m_bTrain;
    bool m_bEarlyUpdate;

    GlobalExamples gExamples;
public:
    Chunker();
    Chunker(bool isTrain);

    ~Chunker();

    void train(ChunkedDataSet &trainGoldSet, InstanceSet &trainSet, ChunkedDataSet &devGoldSet,  InstanceSet &devSet);

    double chunk(InstanceSet &devInstances, ChunkedDataSet &goldDevSet, NNetPara<XPU> &netsPara);
    
private:
    void initTrain(ChunkedDataSet &goldSet, InstanceSet &trainSet);
};

#endif
