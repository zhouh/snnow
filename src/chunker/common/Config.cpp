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

// std::string CConfig::strTrainPath("../../data/chunk/English/test.txt");
// std::string CConfig::strTrainPath("../../data/chunk/English/train.txt");
// std::string CConfig::strDevPath("../../data/chunk/English/test.txt");
// std::string CConfig::strTestPath("../../data/chunk/English/test.txt");

// std::string CConfig::strTrainPath("../../data/chunk/English/single.train");
// std::string CConfig::strDevPath("../../data/chunk/English/single.dev");
// std::string CConfig::strTestPath("../../data/chunk/English/single.test");

int CConfig::nBeamSize = 50;

int CConfig::nWordFeatureNum = 5;
int CConfig::nWordEmbeddingDim = 50;

int CConfig::nCapFeatureNum = 1;
int CConfig::nCapEmbeddingDim = 5;

int CConfig::nPOSFeatureNum = 5;
int CConfig::nPOSEmbeddingDim = 10;

int CConfig::nHiddenSize = 300;

int CConfig::nRound = 1000; 

int CConfig::nBeamBatchSize = 1000;
int CConfig::nGreedyBatchSize = 10000;

int CConfig::nEvaluatePerIters = 20;

int CConfig::nThread = 10;

float CConfig::fRegularizationRate = 1e-8;

float CConfig::fBPRate = 0.01;

float CConfig::fInitRange = 0.1;

float CConfig::fAdaEps = 1e-6;

bool CConfig::bDropOut = true;
float CConfig::fDropoutProb = 0.5;

std::ostream& operator<< (std::ostream &os, const CConfig &config) {
    std::cerr << "train path:\t" << CConfig::strTrainPath << std::endl;
    std::cerr << "dev path:\t" << CConfig::strDevPath << std::endl;
    std::cerr << "test path:\t" << CConfig::strTestPath << std::endl;

    std::cerr << "word feat num:\t" << CConfig::nWordFeatureNum << std::endl;
    std::cerr << "word feat dim:\t" << CConfig::nWordEmbeddingDim << std::endl;
    std::cerr << "POS feat num:\t" << CConfig::nPOSFeatureNum << std::endl;
    std::cerr << "POS feat dim:\t" << CConfig::nPOSEmbeddingDim << std::endl;
    std::cerr << "cap feat num:\t" << CConfig::nCapFeatureNum << std::endl;
    std::cerr << "cap feat dim:\t" << CConfig::nCapEmbeddingDim << std::endl;

    std::cerr << "thread num:\t" << CConfig::nThread << std::endl;

    std::cerr << "beam size:\t" << CConfig::nBeamSize << std::endl;

    std::cerr << "round size:\t" << CConfig::nRound << std::endl;
    std::cerr << "greedybatch size:\t" << CConfig::nGreedyBatchSize << std::endl;
    std::cerr << "beambatch size:\t" << CConfig::nBeamBatchSize << std::endl;
    std::cerr << "hidden size:\t" << CConfig::nHiddenSize << std::endl;
    std::cerr << "regular rate:\t" << CConfig::fRegularizationRate << std::endl;
    std::cerr << "BP rate:\t" << CConfig::fBPRate << std::endl;
    std::cerr << "init range:\t" << CConfig::fInitRange << std::endl;
    std::cerr << "adagrad eps:\t" << CConfig::fAdaEps << std::endl;

    std::cerr << "dropout:\t" << CConfig::bDropOut << std::endl;
    std::cerr << "dropout prob:\t" << CConfig::fDropoutProb << std::endl;
}
