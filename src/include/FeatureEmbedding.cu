#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>

#include "FeatureEmbedding.h"

void FeatureEmbedding::init(const real_t init_range){

    static Random<cpu, real_t> rnd(0);  

    rnd.SampleUniform(&data, -1.0 * init_range, init_range);
}

/**
 * read the pre-train embedding for the feature embdding,
 * We only pre-train the word feature embedding
 */
void FeatureEmbedding::readPreTrain(const std::string &file_name,
                                    const std::tr1::unordered_map<std::string, int> &feature_2_idx){

    std::tr1::unordered_map<std::string, int> pretrain_word_2_idx;
    std::vector<std::vector<real_t>> pretrain_embeddings;
    std::string line;
    std::ifstream in(sFileName);

    int index = 0;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        std::vector<real_t> embeddings;

        std::string word;
        real_t d;
        iss >> word;
        while (iss >> d) {
            embeddings.push_back(d);
        }

        pretrain_embeddings.push_back(embeddings);
        pretrain_word_2_idx[word] = index++;
    }

    std::cerr << "### pre-train words size: " << pretrainEmbs.size() << std::endl;

    // copy the data from cpu to cpu/gpu
    TensorContainer<cpu, 2, real_t> cpu_data(data.shape_);
    Copy(cpu_data, data, data.stream_);

    for (auto& word_index_pair : feature_2_idx) {
        auto pretrain_word_idx_pair = pretrain_word_2_idx.find(word_index_pair.first);

        if (pretrain_word_2_idx.end() != pretrain_word_idx_pair) {  // find it

            int word_index  = word_index_pair.second;
            auto &pre_train_embedding = pretrain_embeddings[pretrain_word_idx_pair->second];

            if (!(word_index >= 0 && word_index < dictionary_size)) {
                std::cerr << "dictSize: " << dictionary_size << std::endl;
                std::cerr << "currWord: " << word_index_pair.first << std::endl;
                std::cerr << "featIndex: " << word_index << std::endl;
            }
            assert (word_index >= 0 && word_index < dictionary_size);
            assert (embedding_dim == static_cast<int>(pre_train_embedding.size()));

            for (int i = 0; i < embedding_dim; i++) {
                cpu_data[word_index][i] = pre_train_embedding[i];
            }
        }
    }

    Copy(data, cpu_data, data.stream_);
}
