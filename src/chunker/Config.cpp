/*************************************************************************
	> File Name: Config.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:49:56 PM CST
 ************************************************************************/
#include "Config.h"

std::string CConfig::strEmbeddingPath("../../data/chunk/English/sen.emb");
std::string CConfig::strTrainPath("../../data/chunk/English/small.train");
std::string CConfig::strDevPath("../../data/chunk/English/small.train");
std::string CConfig::strTestPath("../../data/chunk/English/small.test");

// std::string CConfig::strTrainPath("../../data/chunk/English/train.txt");
// std::string CConfig::strTrainPath("../../data/chunk/English/test.txt");
// std::string CConfig::strDevPath("../../data/chunk/English/test.txt");
// std::string CConfig::strTestPath("../../data/chunk/English/test.txt");

// std::string CConfig::strTrainPath("../../data/chunk/English/single.train");
// std::string CConfig::strDevPath("../../data/chunk/English/single.dev");
// std::string CConfig::strTestPath("../../data/chunk/English/single.test");

int CConfig::nBeamSize = 50;

// int CConfig::nFeatureNum = 6;

// int CConfig::nEmbeddingDim = 50;

int CConfig::nHiddenSize = 300;

int CConfig::nRound = 1000;

int CConfig::nBatchSize = 1000;

int CConfig::nEvaluatePerIters = 20;

int CConfig::nThread = 1;

double CConfig::fRegularizationRate = 1e-8;

double CConfig::fBPRate = 0.1;

double CConfig::fInitRange = 0.1;

double CConfig::fAdaEps = 1e-6;

bool CConfig::bDropOut = false;

