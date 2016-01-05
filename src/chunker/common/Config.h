/*************************************************************************
	> File Name: Config.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 01:49:14 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_CONFIG_H_
#define _CHUNKER_COMMON_CONFIG_H_

#include <iostream>
#include <string>

class CConfig {
public:
    static std::string strTrainPath;
    static std::string strDevPath;
    static std::string strTestPath;
    static std::string strEmbeddingPath;

    static int nBeamSize;

    static int nWordFeatureNum;
    static int nWordEmbeddingDim;
    static int nCapFeatureNum;
    static int nCapEmbeddingDim;
    static int nPOSFeatureNum;
    static int nPOSEmbeddingDim;

    static int nHiddenSize;

    static int nRound;
    static int nBatchSize;
    static int nEvaluatePerIters;
    static int nThread;

    static float fRegularizationRate;
    static float fBPRate;
    static float fInitRange;
    static float fAdaEps;

    static bool bDropOut;

    friend std::ostream& operator<< (std::ostream &os, const CConfig &config);
};

#endif
