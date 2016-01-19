/*************************************************************************
	> File Name: FeatureEmbeddingManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 30 Dec 2015 02:12:04 PM CST
 ************************************************************************/
#include <iostream>
#include <unordered_set>

#include "Config.h"
#include "FeatureEmbeddingManager.h"
#include "FeatureType.h"

void FeatureEmbeddingManager::init(const std::shared_ptr<FeatureManager> &featManagerPtr, const real_t initRange) {
    m_featManagerPtr = featManagerPtr;

    std::vector<FeatureType> featTypes = m_featManagerPtr->getFeatureTypes();
    std::vector<std::shared_ptr<Dictionary>> dictPtrs = m_featManagerPtr->getDictManagerPtrs();

    totalFeatEmbSize = 0;

    for (auto &fType : featTypes) {
        m_lFeatTypes.push_back(fType);
        totalFeatEmbSize += fType.featSize * fType.featEmbSize;
    }

    for (auto &dictPtr : dictPtrs) {
        m_lFeatDictPtrs.push_back(dictPtr);
    }

    m_lEmbeddingNames = m_featManagerPtr->getEmebddingNames();
}

std::vector<std::shared_ptr<FeatureEmbedding>> FeatureEmbeddingManager::loadFeatureEmbeddings(){
    std::vector<std::shared_ptr<FeatureEmbedding>> embeddings(m_lFeatTypes.size());
    std::tr1::unordered_map<std::string, std::shared_ptr<FeatureEmbedding>> name2Embedding;

    std::string line;
    std::string tmp;
    int size;
    std::ifstream is(CConfig::strFeatureEmbeddingManagerPath);
    getline(is, line);
    std::istringstream iss(line);
    iss >> tmp >> size;
    for (int i = 0; i < size; i++) {
        getline(is, line);
        iss.str(line);

        std::string featName;
        int featSize, dictSize, featEmbSize;
        iss >> featName >> featSize >> dictSize >> featEmbSize;
        FeatureType type(featName, featSize, dictSize, featEmbSize);
        const std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

        if (name2Embedding.find(embeddingName) != name2Embedding.end()) {
            std::cerr << "replicated embeddingName: " << embeddingName << std::endl;
            exit(0);
        }

        name2Embedding[embeddingName] = std::make_shared<FeatureEmbedding>(type);
        FeatureEmbedding *fe = name2Embedding[embeddingName].get();

        getline(is, line);
        iss.str(line);
        int dSize, fSize;
        iss >> dSize >> fSize;
        assert (dSize == dictSize);
        assert (fSize == featEmbSize);

        for (int di = 0; di < dSize; di++) {
            getline(is, line);
            iss.str(line);
            for (int ei = 0; ei < fSize; ei++) {
                iss >> fe->data[di][ei];
            }
        }
    }

    for (int i = 0; i < m_lFeatTypes.size(); i++) {
        const FeatureType &type = m_lFeatTypes[i];
        std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

        if (name2Embedding.find(embeddingName) == name2Embedding.end()) {
            std::cerr << "wrong embeddingName: " << embeddingName << std::endl;
            exit(0);
        }

        embeddings[i] = name2Embedding[embeddingName];
    }

    return embeddings;
}

void FeatureEmbeddingManager::saveFeatureEmbeddings(std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbPtrs) {
    assert (m_lFeatTypes.size() == featEmbPtrs.size());

    std::ofstream os(CConfig::strFeatureEmbeddingManagerPath);

    std::unordered_set<std::string> savedEmbNames;

    os << "size" << " " << m_lEmbeddingNames.size() << std::endl;

    for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
        FeatureType &ft = m_lFeatTypes[i];
        std::string embName = m_featManagerPtr->featName2EmbeddingName(ft.typeName);

        if (savedEmbNames.find(embName) != savedEmbNames.end()) {
            continue;
        }

        os << ft;

        FeatureEmbedding *fe = featEmbPtrs[i].get();
        os << fe->dictSize << " " << fe->embeddingSize << std::endl;
        for (int di = 0; di < fe->dictSize; di++) {
            for (int ei = 0; ei < fe->embeddingSize; ei++) {
                os << fe->data[di][ei];

                if (ei == fe->embeddingSize - 1) {
                    os << std::endl;
                } else {
                    os << " ";
                }
            }
        }
    } 
}

std::vector<std::shared_ptr<FeatureEmbedding>> FeatureEmbeddingManager::getInitialzedEmebddings(const real_t initRange) {
    std::vector<std::shared_ptr<FeatureEmbedding>> embeddings(m_lFeatTypes.size());
    std::tr1::unordered_map<std::string, std::shared_ptr<FeatureEmbedding>> name2Embedding;

    for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
        const FeatureType &type = m_lFeatTypes[i];
        const std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

        if (name2Embedding.find(embeddingName) == name2Embedding.end()) {
            name2Embedding[embeddingName] = std::make_shared<FeatureEmbedding>(type);
            name2Embedding[embeddingName]->init(initRange);
        }

        auto found = name2Embedding.find(embeddingName);

        if (m_lFeatTypes[i].typeName == FeatureManager::WORDDESCRIPTION) {
            found->second->readPreTrain(CConfig::strEmbeddingPath, m_lFeatDictPtrs[i]->getWord2IdxMap());
        }
        embeddings[i] = found->second;
    }

    return embeddings;
}

std::vector<std::shared_ptr<FeatureEmbedding>> FeatureEmbeddingManager::getAllZeroEmebddings() {
    std::vector<std::shared_ptr<FeatureEmbedding>> embeddings(m_lFeatTypes.size());
    std::tr1::unordered_map<std::string, std::shared_ptr<FeatureEmbedding>> name2Embedding;

    for (int i = 0; i < static_cast<int>(m_lFeatTypes.size()); i++) {
        const FeatureType &type = m_lFeatTypes[i];
        const std::string embeddingName = m_featManagerPtr->featName2EmbeddingName(type.typeName);

        if (name2Embedding.find(embeddingName) == name2Embedding.end()) {
            name2Embedding[embeddingName] = std::make_shared<FeatureEmbedding>(type);
        }

        auto found = name2Embedding.find(embeddingName);

        embeddings[i] = found->second;
    }

    return embeddings;
}

void FeatureEmbeddingManager::returnInput(std::vector<FeatureVector> &featVecs, std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs, TensorContainer<cpu, 2, real_t> &input){
    // TODO: if neccessary ?
	// initialize the input
	// input.Resize( Shape2( beamSize, totalFeatEmbSize ), static_cast<real_t>(0.0));

	for(unsigned beamIndex = 0; beamIndex < static_cast<unsigned>(featVecs.size()); beamIndex++) { // for every beam item

        FeatureVector &featVector = featVecs[beamIndex];

		int inputIndex = 0;
        for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featVector.size()); featTypeIndex++) {
            const std::vector<int> &curFeatVec = featVector[featTypeIndex];
            const int curFeatSize = m_lFeatTypes[featTypeIndex].featSize;
            const int curEmbSize  = m_lFeatTypes[featTypeIndex].featEmbSize;
            std::shared_ptr<FeatureEmbedding> &curFeatEmb = featEmbs[featTypeIndex];

            for (auto featId : curFeatVec) {
                for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                    // std::cerr << "[data]dim0 size: " << curFeatEmb->data.shape_[0] << std::endl;
                    // std::cerr << "featId: " << featId << std::endl;
                    // std::cerr << "[data]dim1 size: " << curFeatEmb->data.shape_[1] << std::endl;
                    // std::cerr << "embIndex: " << embIndex << std::endl;
                    // std::cerr << "[input]dim0 size: " << input.shape_[0] << std::endl;
                    // std::cerr << "beamIndex: " << beamIndex << std::endl;
                    // std::cerr << "[input]dim1 size: " << input.shape_[1] << std::endl;
                    // std::cerr << "inputIndex: " << inputIndex << std::endl;
                    input[beamIndex][inputIndex++] = curFeatEmb->data[featId][embIndex];
                    // curFeatEmb->data[featId][embIndex];
                }
            }
            // for (int featureIndex = 0; featureIndex < curFeatSize; featureIndex++) {
            //     for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
            //         input[beamIndex][inputIndex++] = curFeatEmb->data[curFeatVec[featureIndex]][embIndex];
            //     }
            // }
        }
    }
}
