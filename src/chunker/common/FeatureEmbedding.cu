#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>

#include "FeatureEmbedding.h"
void FeatureEmbedding::init(const real_t initRange){
    Random<EMBEDDING_XPU, real_t> rnd(0);  

    rnd.SampleUniform(&data, -1.0 * initRange, initRange);
}

void FeatureEmbedding::readPreTrain(const std::string &sFileName, const std::tr1::unordered_map<std::string, int> &word2Idx){
    std::tr1::unordered_map<std::string, int> pretrainWords;
    std::vector<std::vector<real_t>> pretrainEmbs;
    std::string line;
    std::ifstream in(sFileName);

    int index = 0;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        std::vector<real_t> embedding;

        std::string word;
        real_t d;
        iss >> word;
        while (iss >> d) {
            embedding.push_back(d);
        }

        pretrainEmbs.push_back(embedding);
        pretrainWords[word] = index++;
    }

    std::cerr << "  pretrainWords's size: " << pretrainEmbs.size() << std::endl;

    TensorContainer<cpu, 2, real_t> cpu_data(data.shape_);
    Copy(cpu_data, data, data.stream_);

    for (auto &wordPair : word2Idx) {
        auto ret = pretrainWords.find(wordPair.first);

        if (pretrainWords.end() != ret) {
            int featIndex  = wordPair.second;
            auto &preTrain = pretrainEmbs[ret->second];

            if (!(featIndex >= 0 && featIndex < dictSize)) {
                std::cerr << "dictSize: " << dictSize << std::endl;
                std::cerr << "currWord: " << wordPair.first << std::endl;
                std::cerr << "featIndex: " << featIndex << std::endl;
            }
            assert (featIndex >= 0 && featIndex < dictSize);
            assert (embeddingSize == static_cast<int>(preTrain.size()));

            for (int i = 0; i < embeddingSize; i++) {
                cpu_data[featIndex][i] = preTrain[i];
            }
        }
    }

    Copy(data, cpu_data, data.stream_);
}
