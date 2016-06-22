/*************************************************************************
	> File Name: GreedyChunker.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 20 Jun 2016 06:50:23 PM CST
 ************************************************************************/
#ifndef SNNOW_GREEDYCHUNKER_H
#define SNNOW_GREEDYCHUNKER_H

#include "ChunkerFeatureExtractor.h"
#include "ChunkerTransitionSystem.h"
#include "SeqLabelerDataSet.h"
#include "base/TrainingExample.h"
#include "SeqLabelerInput.h"
#include "SeqLabelerOutput.h"

#ifdef DEBUG
// #define DEBUG1
#endif // DEBUG

class GreedyChunker {
public:
    GreedyChunker() = default;
    ~GreedyChunker() = default;

    // void generateGreedyTrainingExamples(DepArcStandardSystem* transit_system,
    //                                     DepParseDataSet& training_data,
    //                                     std::vector<std::shared_ptr<Example>> & examples);
    void generateTrainingExamples(ChunkerTransitionSystem *transit_system_ptr,
                                  ChunkerFeatureExtractor *feature_extractor_ptr,
                                  SeqLabelerDataSet *training_data_ptr,
                                  std::vector<std::shared_ptr<Example>> &example_ptrs) {
        example_ptrs.clear();

        for (int i = 0; i < training_data_ptr->getSize(); i++) {
            SeqLabelerInput *input_ptr = static_cast<SeqLabelerInput*>(training_data_ptr->inputs[i]);
            SeqLabelerOutput *output_ptr = static_cast<SeqLabelerOutput*>(training_data_ptr->outputs[i]);
            RawSequence &raw_seq = training_data_ptr->raw_sequences_[i];

            feature_extractor_ptr->generateCache(input_ptr, output_ptr, raw_seq);

            std::shared_ptr<ChunkerState> state_ptr(new ChunkerState());
            state_ptr->setSequenceInput(input_ptr);
            state_ptr->previous = state_ptr.get();
            state_ptr->setSequenceLength(raw_seq.size());

#ifdef DEBUG1
            std::cout << raw_seq;
            int index = 0;
#endif // DEBUG1

            while (!state_ptr->complete()) {
#ifdef  DEBUG1
                std::cerr << "index: " << index << std::endl;
#endif // DEBUG1
                std::vector<int> labels;

                FeatureVector fv = feature_extractor_ptr->getFeatureVectors(state_ptr.get(), input_ptr);

#ifdef DEBUG1
                std::cout << "feature vector:" << std::endl;
                for (int fi = 0; fi < fv.size(); fi++) {
                    auto &v = fv[fi];

                    std::cout << "\t";
                    for (int f : v) {
                        std::cout << f << ":" << feature_extractor_ptr->dictionary_ptrs_table_[fi]->getString(f) << "\t";
                    }
                    std::cout << std::endl;
                }
#endif // DEBUG1
                transit_system_ptr->getValidActs(state_ptr.get(), labels);
#ifdef DEBUG1
                std::cout << "valid labels:" << std::endl;
                std::cout << "\t";
                for (int li = 0; li < labels.size(); li++) {
                    int l = labels[li];
                    std::cout << li << ":" << feature_extractor_ptr->dictionary_ptrs_table_[feature_extractor_ptr->c_label_dict_index_]->getString(li) << "\t";
                }
                std::cout << std::endl;
#endif // DEBUG1

                Action *gold_act_ptr = transit_system_ptr->StandardMove(state_ptr.get(), output_ptr);
                const int gold_act_id = gold_act_ptr->getActionCode();

#ifdef DEBUG1
                std::cout << "gold_act_id: " << gold_act_id << std::endl;
#endif // DEBUG1

#ifdef  DEBUG1
                std::cout << "state(before move):" << std::endl;
                state_ptr->print("\t");
#endif // DEBUG1
                transit_system_ptr->Move(state_ptr.get(), gold_act_ptr);
#ifdef  DEBUG1
                std::cout << "state(after move):" << std::endl;
                state_ptr->print("\t");
#endif // DEBUG1

                labels[gold_act_id] = 1;
                std::shared_ptr<Example> example_ptr(new Example(fv, labels));
                example_ptrs.push_back(example_ptr);
            }
        }
    }

private:
    GreedyChunker(const GreedyChunker&) = delete;
    GreedyChunker&operator=(const GreedyChunker&) = delete;
};

#endif // SNNOW_GREEDYCHUNKER_H
