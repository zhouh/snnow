/*************************************************************************
	> File Name: Config.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:49:56 PM CST
 ************************************************************************/
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <cctype>
#include <algorithm>

#include "Config.h"

std::string CConfig::strEmbeddingPath("../../data/chunk/English/sen.emb");

// std::string CConfig::strTrainPath("../../data/chunk/English/small.train");
// std::string CConfig::strDevPath("../../data/chunk/English/small.train");
// std::string CConfig::strTestPath("../../data/chunk/English/small.test");

// std::string CConfig::strTrainPath("../../data/chunk/English/test.txt");
std::string CConfig::strTrainPath("../../data/chunk/English/train.txt");
std::string CConfig::strDevPath("../../data/chunk/English/test.txt");
std::string CConfig::strTestPath("../../data/chunk/English/test.txt");

// std::string CConfig::strTrainPath("../../data/chunk/English/single.train");
// std::string CConfig::strDevPath("../../data/chunk/English/single.dev");
// std::string CConfig::strTestPath("../../data/chunk/English/single.test");

int CConfig::nBeamSize = 50;
int CConfig::nGPUBatchSize = 50;

int CConfig::nWordFeatureNum = 5;
int CConfig::nWordEmbeddingDim = 50;

int CConfig::nCapFeatureNum = 1;
int CConfig::nCapEmbeddingDim = 5;

int CConfig::nPOSFeatureNum = 5;
int CConfig::nPOSEmbeddingDim = 10;

int CConfig::nLabelFeatureNum = 2;
int CConfig::nLabelEmbeddingDim = 5;

int CConfig::nHiddenSize = 300;

int CConfig::nRound = 2000; 

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

bool CConfig::bFineTune = true;

std::ostream& operator<< (std::ostream &os, const CConfig &config) {
    std::cerr << "train path:       " << CConfig::strTrainPath << std::endl;
    std::cerr << "dev path:         " << CConfig::strDevPath << std::endl;
    std::cerr << "test path:        " << CConfig::strTestPath << std::endl;

    std::cerr << "word feat num:    " << CConfig::nWordFeatureNum << std::endl;
    std::cerr << "word feat dim:    " << CConfig::nWordEmbeddingDim << std::endl;
    std::cerr << "POS feat num:     " << CConfig::nPOSFeatureNum << std::endl;
    std::cerr << "POS feat dim:     " << CConfig::nPOSEmbeddingDim << std::endl;
    std::cerr << "cap feat num:     " << CConfig::nCapFeatureNum << std::endl;
    std::cerr << "cap feat dim:     " << CConfig::nCapEmbeddingDim << std::endl;
    std::cerr << "label feat num:   " << CConfig::nLabelFeatureNum << std::endl;
    std::cerr << "label feat dim:   " << CConfig::nLabelEmbeddingDim << std::endl;

    std::cerr << "thread num:       " << CConfig::nThread << std::endl;

    std::cerr << "beam size:        " << CConfig::nBeamSize << std::endl;

    std::cerr << "round size:       " << CConfig::nRound << std::endl;
    std::cerr << "greedybatch size: " << CConfig::nGreedyBatchSize << std::endl;
    std::cerr << "beambatch size:   " << CConfig::nBeamBatchSize << std::endl;
    std::cerr << "hidden size:      " << CConfig::nHiddenSize << std::endl;
    std::cerr << "regular rate:     " << CConfig::fRegularizationRate << std::endl;
    std::cerr << "BP rate:          " << CConfig::fBPRate << std::endl;
    std::cerr << "init range:       " << CConfig::fInitRange << std::endl;
    std::cerr << "adagrad eps:      " << CConfig::fAdaEps << std::endl;

    std::cerr << "dropout:          " << CConfig::bDropOut << std::endl;
    std::cerr << "dropout prob:     " << CConfig::fDropoutProb << std::endl;
}

inline void my_trim(std::string &s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        s = "";
        return ;
    }
    auto end = s.find_last_not_of(" \t\n\r");

    s = s.substr(start, end - start + 1);
}

void CConfig::readConfiguration(const std::string &configPath) {
    using namespace std;

    unordered_set<string> attributes;
    attributes.insert("strTrainPath");
    attributes.insert("strDevPath");
    attributes.insert("strTestPath");
    attributes.insert("strEmbeddingPath");
    attributes.insert("nBeamSize");
    attributes.insert("nGPUBatchSize");
    attributes.insert("nWordFeatureNum");
    attributes.insert("nWordEmbeddingDim");
    attributes.insert("nCapFeatureNum");
    attributes.insert("nCapEmbeddingDim");
    attributes.insert("nPOSFeatureNum");
    attributes.insert("nPOSEmbeddingDim");
    attributes.insert("nLabelFeatureNum");
    attributes.insert("nLabelEmbeddingDim");
    attributes.insert("nHiddenSize");
    attributes.insert("nRound");
    attributes.insert("nGreedyBatchSize");
    attributes.insert("nBeamBatchSize");
    attributes.insert("nEvaluatePerIters");
    attributes.insert("nThread");
    attributes.insert("fRegularizationRate");
    attributes.insert("fBPRate");
    attributes.insert("fInitRange");
    attributes.insert("fAdaEps");
    attributes.insert("bDropOut");
    attributes.insert("fDropoutProb");
    attributes.insert("bFineTune");

    std::ifstream is(configPath);
    while (!is.eof()) {
        pair<string, string> att;

        std::string line;
        getline(is, line);
        my_trim(line);
        while (!is.eof() && (line == "" || line[0] == '#')) {
            getline(is, line);
            my_trim(line);
        }
        if (is.eof()) {
            break;
        }
        std::size_t found = line.find(":");
        std::string key = line.substr(0, found);
        my_trim(key);
        std::string value = line.substr(found + 1);
        my_trim(value);

        att =  std::make_pair(key, value);

        auto argumentIsWrong = [&att](){
            std::cerr << "argument " << att.first << ": " << att.second << " is wrong!" << std::endl;
            exit(0);
        };

        if (attributes.find(att.first) == attributes.end()) {
            argumentIsWrong();
        }

        if (att.first == "strTrainPath") {
            strTrainPath = att.second;
        } else if (att.first == "strDevPath") {
            strDevPath = att.second;
        } else if (att.first == "strTestPath") {
            strTestPath = att.second;
        } else if (att.first == "strEmbeddingPath") {
            strEmbeddingPath = att.second;
        } else if (att.first == "nBeamSize") {
            nBeamSize = stoi(att.second);
        } else if (att.first == "nGPUBatchSize") {
            nGPUBatchSize = stoi(att.second);
        } else if (att.first == "nWordFeatureNum") {
            nWordFeatureNum = stoi(att.second);
        } else if (att.first == "nWordEmbeddingDim") {
            nWordEmbeddingDim = stoi(att.second);
        } else if (att.first == "nCapFeatureNum") {
            nCapFeatureNum = stoi(att.second);
        } else if (att.first == "nCapEmbeddingDim") {
            nCapEmbeddingDim = stoi(att.second);
        } else if (att.first == "nPOSFeatureNum") {
            nPOSFeatureNum = stoi(att.second);
        } else if (att.first == "nPOSEmbeddingDim") {
            nPOSEmbeddingDim = stoi(att.second);
        } else if (att.first == "nLabelFeatureNum") {
            nLabelFeatureNum = stoi(att.second);
        } else if (att.first == "nLabelEmbeddingDim") {
            nLabelEmbeddingDim = stoi(att.second);
        } else if (att.first == "nHiddenSize") {
            nHiddenSize = stoi(att.second);
        } else if (att.first == "nRound") {
            nRound = stoi(att.second);
        } else if (att.first == "nGreedyBatchSize") {
            nGreedyBatchSize = stoi(att.second);
        } else if (att.first == "nBeamBatchSize") {
            nBeamBatchSize = stoi(att.second);
        } else if (att.first == "nEvaluatePerIters") {
            nEvaluatePerIters = stoi(att.second);
        } else if (att.first == "nThread") {
            nThread = stoi(att.second);
        } else if (att.first == "fRegularizationRate") {
            fRegularizationRate = stof(att.second);
        } else if (att.first == "fBPRate") {
            fBPRate = stof(att.second);
        } else if (att.first == "fInitRange") {
            fInitRange = stof(att.second);
        } else if (att.first == "fAdaEps") {
            fAdaEps = stof(att.second);
        } else if (att.first == "bDropOut") {
            if (att.second == "true") {
                bDropOut = true;
            } else {
                bDropOut = false;
            }
        } else if (att.first == "fDropoutProb") {
            fDropoutProb = stof(att.second);
        } else if (att.first == "nFineTune") {
            if (att.second == "true") {
                bFineTune = true;
            } else {
                bFineTune = false;
            }
        }
    }

    if (nGreedyBatchSize % nGPUBatchSize != 0) {
        std::cerr << "nGreedyBatchSize: " << nGreedyBatchSize << " should by divisible by nGPUBatchSize: " << nGPUBatchSize << std::endl;
        exit(0);
    }
    if (nGreedyBatchSize < nGPUBatchSize * nThread) {
        std::cerr << "nGreedyBatchSize: " << nGreedyBatchSize << " should be more than nGPUBatchSize * nThread: " << nGPUBatchSize << " * " << nThread << ")" << std::endl;
        exit(0);
    }
}

void CConfig::saveConfiguration(const std::string &configPath) {

}
