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
#include "nets/Model.h"
#include "nets/FeedForwardNNet.h"

DECLARE_int32(hidden_size);

#ifdef DEBUG
// #define DEBUG1
// #define DEBUG2
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

    std::vector<std::vector<std::string>> predictForDataSet(ChunkerTransitionSystem *transit_system_ptr,
                                      ChunkerFeatureExtractor *feature_extractor_ptr,
                                      Model<gpu> *model_ptr,
                                      SeqLabelerDataSet *data_ptr) {
        std::vector<std::vector<std::string>> ret_label_seq_set;

        const int num_in = feature_extractor_ptr->getTotalInputSize();
        const int num_hidden = FLAGS_hidden_size;
        const int num_out = transit_system_ptr->action_factory_ptr_->total_action_num;

        for (int i = 0; i < data_ptr->getSize(); i++) {
            std::vector<std::string> label_seq;

            SeqLabelerInput *input_ptr = static_cast<SeqLabelerInput *>(data_ptr->inputs[i]);
            SeqLabelerOutput *output_ptr = static_cast<SeqLabelerOutput *>(data_ptr->outputs[i]);
            RawSequence &raw_seq = data_ptr->raw_sequences_[i];

            feature_extractor_ptr->generateCache(input_ptr, output_ptr, raw_seq);

            std::shared_ptr<ChunkerState> state_ptr(new ChunkerState());
            state_ptr->setSequenceInput(input_ptr);
            state_ptr->previous = state_ptr.get();
            state_ptr->setSequenceLength(raw_seq.size());

            int index = 0;
            TensorContainer<cpu, 2, real_t> input(Shape2(1, num_in));
            TensorContainer<cpu, 2, real_t> predict_output(Shape2(1, num_out));

            //while (!state_ptr->complete()) {
            while (index < raw_seq.size()) {
                index++;
#ifdef  DEBUG2
                std::cerr << "index: " << index << std::endl;
#endif // DEBUG2
                std::shared_ptr<FeedForwardNNet<gpu>> net_ptr;
                net_ptr.reset(new FeedForwardNNet<gpu>(1, num_in, num_hidden, num_out, model_ptr));

                input = 0;
                predict_output = 0;
                std::vector<int> labels;

                // FeatureVector fv = feature_extractor_ptr->getFeatureVectors(state_ptr.get(), input_ptr);
                FeatureVectors fvs;
                fvs.push_back(feature_extractor_ptr->getFeatureVectors(state_ptr.get(), input_ptr));

                transit_system_ptr->getValidActs(state_ptr.get(), labels);

                feature_extractor_ptr->returnInput(fvs, model_ptr->featEmbs, input);
// /*

                net_ptr->ChunkForward(input, predict_output, false);
                const int insti = 0;
                int opt_act = -1;
                std::vector<int> &valid_acts = labels;

                for (int i = 0; i < valid_acts.size(); i++) {
                    if (valid_acts[i] >= 0) {
                        if (opt_act == -1 || predict_output[insti][i] > predict_output[insti][opt_act]) {
                            opt_act = i;
                        }
                    }
                }
                label_seq.push_back(feature_extractor_ptr->getLabelString(opt_act));

#ifdef DEBUG2
                std::cout << "feature vector:" << std::endl;
                for (int fi = 0; fi < fv.size(); fi++) {
                    std::vector<int> v = fv.getVector(fi);

                    std::cout << "\t";
                    for (int f : v) {
                        std::cout << f << ":" << feature_extractor_ptr->dictionary_ptrs_table_[fi]->getString(f) << "\t";
                    }
                    std::cout << std::endl;
                }
#endif // DEBUG2
#ifdef DEBUG2
                std::cout << "valid labels:" << std::endl;
                std::cout << "\t";
                for (int li = 0; li < labels.size(); li++) {
                    int l = labels[li];
                    std::cout << li << ":" << feature_extractor_ptr->dictionary_ptrs_table_[feature_extractor_ptr->c_label_dict_index_]->getString(li) << "(" << l << ")" << "\t";
                }
                std::cout << std::endl;
#endif // DEBUG2

#ifdef  DEBUG2
                std::cout << "state(before move):" << std::endl;
                state_ptr->print("\t");
#endif // DEBUG2

                transit_system_ptr->Move(state_ptr.get(), ChunkerActionFactory::action_table[opt_act].get());

#ifdef  DEBUG2
                std::cout << "state(after move):" << std::endl;
                state_ptr->print("\t");
                std::cerr << "While end..." << std::endl;
#endif // DEBUG2
// */
            }

            ret_label_seq_set.push_back(label_seq);
        }

        return ret_label_seq_set;
    }

private:
    GreedyChunker(const GreedyChunker&) = delete;
    GreedyChunker&operator=(const GreedyChunker&) = delete;
};

#endif // SNNOW_GREEDYCHUNKER_H
