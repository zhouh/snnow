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

bool CConfig::loadModel = false;
bool CConfig::saveModel = false;

std::string CConfig::strDictManagerPath;
std::string CConfig::strNetModelPath;
std::string CConfig::strFeatureEmbeddingManagerPath;
std::string CConfig::strFeatureManagerPath;
std::string CConfig::strActionStandardSystemPath;

std::string CConfig::strEmbeddingPath("../../data/chunk/English/sen.emb");

std::string CConfig::strWordTablePath("../../data/chunk/English/giga.dict");

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

int CConfig::nChunkWordFeatureNum = 4;
int CConfig::nChunkWordEmbeddingDim = 50;

int CConfig::nChunkPOSFeatureNum = 4;
int CConfig::nChunkPOSEmbeddingDim = 10;

int CConfig::nHiddenSize = 300;

int CConfig::nRound = 2000; 

int CConfig::nBeamBatchSize = 1000;
int CConfig::nBeamBatchDecoderItemSize = 10;
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
    std::cerr << "embedding path:   " << CConfig::strEmbeddingPath << std::endl;
    std::cerr << "word table path:  " << CConfig::strWordTablePath << std::endl;
    std::cerr << "train path:       " << CConfig::strTrainPath << std::endl;
    std::cerr << "dev path:         " << CConfig::strDevPath << std::endl;
    std::cerr << "test path:        " << CConfig::strTestPath << std::endl;

    std::cerr << "word feat num:      " << CConfig::nWordFeatureNum << std::endl;
    std::cerr << "word feat dim:      " << CConfig::nWordEmbeddingDim << std::endl;
    std::cerr << "POS feat num:       " << CConfig::nPOSFeatureNum << std::endl;
    std::cerr << "POS feat dim:       " << CConfig::nPOSEmbeddingDim << std::endl;
    std::cerr << "cap feat num:       " << CConfig::nCapFeatureNum << std::endl;
    std::cerr << "cap feat dim:       " << CConfig::nCapEmbeddingDim << std::endl;
    std::cerr << "label feat num:     " << CConfig::nLabelFeatureNum << std::endl;
    std::cerr << "label feat dim:     " << CConfig::nLabelEmbeddingDim << std::endl;
    std::cerr << "chunkword feat num: " << CConfig::nChunkWordFeatureNum << std::endl;
    std::cerr << "chunkword feat dim: " << CConfig::nChunkWordEmbeddingDim << std::endl;
    std::cerr << "chunkPOS feat num:  " << CConfig::nChunkPOSFeatureNum << std::endl;
    std::cerr << "chunkPOS feat dim:  " << CConfig::nChunkPOSEmbeddingDim << std::endl;

    std::cerr << "thread num:         " << CConfig::nThread << std::endl;

    std::cerr << "beam size:          " << CConfig::nBeamSize << std::endl;

    std::cerr << "round size:         " << CConfig::nRound << std::endl;
    std::cerr << "greedybatch size:   " << CConfig::nGreedyBatchSize << std::endl;
    std::cerr << "beambatch size:     " << CConfig::nBeamBatchSize << std::endl;
    std::cerr << "decoderitem size:   " << CConfig::nBeamBatchDecoderItemSize << std::endl;
    std::cerr << "hidden size:        " << CConfig::nHiddenSize << std::endl;
    std::cerr << "regular rate:       " << CConfig::fRegularizationRate << std::endl;
    std::cerr << "BP rate:            " << CConfig::fBPRate << std::endl;
    std::cerr << "init range:         " << CConfig::fInitRange << std::endl;
    std::cerr << "adagrad eps:        " << CConfig::fAdaEps << std::endl;

    std::cerr << "dropout:            " << CConfig::bDropOut << std::endl;
    std::cerr << "dropout prob:       " << CConfig::fDropoutProb << std::endl;

    std::cerr << "fine-tune:          " << CConfig::bFineTune << std::endl;
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
    attributes.insert("loadModel");
    attributes.insert("saveModel");
    attributes.insert("strDictManagerPath");
    attributes.insert("strNetModelPath");
    attributes.insert("strFeatureEmbeddingManagerPath");
    attributes.insert("strFeatureManagerPath");
    attributes.insert("strActionStandardSystemPath");
    attributes.insert("strTrainPath");
    attributes.insert("strWordTablePath");
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
    attributes.insert("nChunkWordFeatureNum");
    attributes.insert("nChunkWordEmbeddingDim");
    attributes.insert("nChunkPOSFeatureNum");
    attributes.insert("nChunkPOSEmbeddingDim");
    attributes.insert("nHiddenSize");
    attributes.insert("nRound");
    attributes.insert("nGreedyBatchSize");
    attributes.insert("nBeamBatchSize");
    attributes.insert("nBeamBatchDecoderItemSize");
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

        if (att.first == "loadModel") {
            if (att.second == "true") {
                CConfig::loadModel = true;
            } else {
                CConfig::loadModel = false;
            }
        } else if (att.first == "saveModel") {
            if (att.second == "true") {
                CConfig::saveModel = true;
            } else {
                CConfig::saveModel = false;
            }
        } else if (att.first == "strDictManagerPath") {
            CConfig::strDictManagerPath = att.second;
        } else if (att.first == "strNetModelPath") {
            CConfig::strNetModelPath = att.second;
        } else if (att.first == "strFeatureEmbeddingManagerPath") {
            CConfig::strFeatureEmbeddingManagerPath = att.second;
        } else if (att.first == "strFeatureManagerPath") {
            CConfig::strFeatureManagerPath = att.second;
        } else if (att.first == "strActionStandardSystemPath") {
            CConfig::strActionStandardSystemPath = att.second;
        } else if (att.first == "strTrainPath") {
            CConfig::strTrainPath = att.second;
        } else if (att.first == "strWordTablePath") {
            CConfig::strWordTablePath = att.second;
        } else if (att.first == "strDevPath") {
            CConfig::strDevPath = att.second;
        } else if (att.first == "strTestPath") {
            CConfig::strTestPath = att.second;
        } else if (att.first == "strEmbeddingPath") {
            CConfig::strEmbeddingPath = att.second;
        } else if (att.first == "nBeamSize") {
            CConfig::nBeamSize = stoi(att.second);
        } else if (att.first == "nGPUBatchSize") {
            CConfig::nGPUBatchSize = stoi(att.second);
        } else if (att.first == "nWordFeatureNum") {
            CConfig::nWordFeatureNum = stoi(att.second);
        } else if (att.first == "nWordEmbeddingDim") {
            CConfig::nWordEmbeddingDim = stoi(att.second);
        } else if (att.first == "nCapFeatureNum") {
            CConfig::nCapFeatureNum = stoi(att.second);
        } else if (att.first == "nCapEmbeddingDim") {
            CConfig::nCapEmbeddingDim = stoi(att.second);
        } else if (att.first == "nPOSFeatureNum") {
            CConfig::nPOSFeatureNum = stoi(att.second);
        } else if (att.first == "nPOSEmbeddingDim") {
            CConfig::nPOSEmbeddingDim = stoi(att.second);
        } else if (att.first == "nLabelFeatureNum") {
            CConfig::nLabelFeatureNum = stoi(att.second);
        } else if (att.first == "nLabelEmbeddingDim") {
            CConfig::nLabelEmbeddingDim = stoi(att.second);
        } else if (att.first == "nChunkWordFeatureNum") {
            CConfig::nChunkWordFeatureNum = stoi(att.second);
        } else if (att.first == "nChunkWordEmbeddingDim") {
            CConfig::nChunkWordEmbeddingDim = stoi(att.second);
        } else if (att.first == "nChunkPOSFeatureNum") {
            CConfig::nChunkPOSFeatureNum = stoi(att.second);
        } else if (att.first == "nChunkPOSEmbeddingDim") {
            CConfig::nChunkPOSEmbeddingDim = stoi(att.second);
        } else if (att.first == "nHiddenSize") {
            CConfig::nHiddenSize = stoi(att.second);
        } else if (att.first == "nRound") {
            CConfig::nRound = stoi(att.second);
        } else if (att.first == "nGreedyBatchSize") {
            CConfig::nGreedyBatchSize = stoi(att.second);
        } else if (att.first == "nBeamBatchSize") {
            CConfig::nBeamBatchSize = stoi(att.second);
        } else if (att.first == "nBeamBatchDecoderItemSize") {
            CConfig::nBeamBatchDecoderItemSize = stoi(att.second);
        } else if (att.first == "nEvaluatePerIters") {
            CConfig::nEvaluatePerIters = stoi(att.second);
        } else if (att.first == "nThread") {
            CConfig::nThread = stoi(att.second);
        } else if (att.first == "fRegularizationRate") {
            CConfig::fRegularizationRate = stof(att.second);
        } else if (att.first == "fBPRate") {
            CConfig::fBPRate = stof(att.second);
        } else if (att.first == "fInitRange") {
            CConfig::fInitRange = stof(att.second);
        } else if (att.first == "fAdaEps") {
            CConfig::fAdaEps = stof(att.second);
        } else if (att.first == "bDropOut") {
            if (att.second == "true") {
                CConfig::bDropOut = true;
            } else {
                CConfig::bDropOut = false;
            }
        } else if (att.first == "fDropoutProb") {
            CConfig::fDropoutProb = stof(att.second);
        } else if (att.first == "bFineTune") {
            if (att.second == "true") {
                CConfig::bFineTune = true;
            } else {
                CConfig::bFineTune = false;
            }
        }
    }
    if (CConfig::nGreedyBatchSize % CConfig::nGPUBatchSize != 0) {
        std::cerr << "nGreedyBatchSize: " << nGreedyBatchSize << " should be divisible by nGPUBatchSize: " << nGPUBatchSize << std::endl;
        exit(0);
    }

    if (CConfig::nBeamBatchSize % CConfig::nThread != 0) {
        std::cerr << "nBeamBatchSize: " << CConfig::nBeamBatchSize << " should be divisible by nThread: " << CConfig::nThread << std::endl;
        exit(0);
    }

    if ((CConfig::nBeamBatchSize / CConfig::nThread) % CConfig::nBeamBatchDecoderItemSize != 0) {
        std::cerr << "(nBeamBatchSize / nThread): " << (CConfig::nBeamBatchSize / CConfig::nThread) << " should be divisible by nBeamBatchDecoderItemSize: " << CConfig::nBeamBatchDecoderItemSize << std::endl;
        exit(0);
    }

    if (CConfig::nGreedyBatchSize < CConfig::nGPUBatchSize * CConfig::nThread) {
        std::cerr << "nGreedyBatchSize: " << nGreedyBatchSize << " should be more than nGPUBatchSize * nThread: (" << CConfig::nGPUBatchSize << " * " << CConfig::nThread << ")" << std::endl;
        exit(0);
    }
}

void CConfig::saveConfiguration(const std::string &configPath) {

}
