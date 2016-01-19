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
    static bool loadModel;
    static bool saveModel;

    static std::string strModelDirPath;

    static std::string strEmbeddingPath;

    static std::string strWordTablePath;

    static std::string strTrainPath;
    static std::string strDevPath;
    static std::string strTestPath;

    static int nBeamSize;
    static int nGPUBatchSize;

    static int nWordFeatureNum;
    static int nWordEmbeddingDim;
    static int nCapFeatureNum;
    static int nCapEmbeddingDim;
    static int nPOSFeatureNum;
    static int nPOSEmbeddingDim;
    static int nLabelFeatureNum;
    static int nLabelEmbeddingDim;

    static int nHiddenSize;

    static int nRound;
    static int nGreedyBatchSize;
    static int nBeamBatchSize;
    static int nBeamBatchDecoderItemSize;
    static int nEvaluatePerIters;
    static int nSaveModelPerIters;
    static int nThread;

    static float fRegularizationRate;
    static float fBPRate;
    static float fInitRange;
    static float fAdaEps;

    static bool bDropOut;
    static float fDropoutProb;

    static bool bFineTune;
    static bool bReadPretrain;

    static void readConfiguration(const std::string &configPath);
    static void saveConfiguration(const std::string &configPath);
    friend std::ostream& operator<< (std::ostream &os, const CConfig &config);

private:
    static std::pair<std::string, std::string> getNextAttribute(std::istream &is);
};

#endif
