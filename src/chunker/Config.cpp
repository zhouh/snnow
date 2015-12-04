/*************************************************************************
	> File Name: Config.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 01:57:44 PM CST
 ************************************************************************/
#include "Config.h"

string CConfig::strEmbeddingPath("../../data/chunk/English/sen.emb");
string CConfig::strTrainPath("../../data/chunk/English/small.train");
string CConfig::strDevPath("../../data/chunk/English/small.train");
string CConfig::strTestPath("../../data/chunk/English/small.test");

//string CConfig::strTrainPath("../../data/chunk/English/train.txt");
//string CConfig::strTestPath("../../data/chunk/English/test.txt");

int CConfig::nBeamSize = 50;

int CConfig::nFeatureNum = 1;

int CConfig::nEmbeddingDim = 50;

int CConfig::nHiddenSize = 200;

int CConfig::nRound = 100;

int CConfig::nBatchSize = 10;

int CConfig::nEvaluatePerIters = 10;

int CConfig::nThread = 1;

double CConfig::fRegularizationRate = 1e-8;

double CConfig::fBPRate = 0.1;

double CConfig::fInitRange = 0.1;

double CConfig::fAdaEps = 1e-6;

bool CConfig::bDropOut = false;
