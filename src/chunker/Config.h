/*************************************************************************
	> File Name: Config.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 01:49:14 PM CST
 ************************************************************************/
#ifndef _CHUNKER_CONFIG_H_
#define _CHUNKER_CONFIG_H_

#include <string>

using std::string;

class CConfig {
public:
    static string strTrainPath;
    static string strDevPath;
    static string strTestPath;
    static string strEmbeddingPath;

    static int nBeamSize;

    static int nFeatureNum;
    static int nEmbeddingDim;

    static int nHiddenSize;

    static int nRound;
    static int nBatchSize;
    static int nEvaluatePerIters;
    static int nThread;

    static double fRegularizationRate;
    static double fBPRate;
    static double fInitRange;
    static double fAdaEps;

    static bool bDropOut;
};

#endif
