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

class GreedyChunker {
public:
    GreedyChunker() = default;
    ~GreedyChunker() = default;

    // void generateGreedyTrainingExamples(DepArcStandardSystem* transit_system,
    //                                     DepParseDataSet& training_data,
    //                                     std::vector<std::shared_ptr<Example>> & examples);
    void generateTrainingExamples(ChunkerTransitionSystem &transit_system,
                                  ChunkerFeatureExtractor &feature_extractor,
                                  SeqLabelerDataSet &training_data,
                                  std::vector<std::shared_ptr<Example>> &example_ptrs) {
        example_ptrs.clear();

        for (int i = 0; i < training_data.getSize(); i++) {
            SeqLabelerInput *input = static_cast<SeqLabelerInput&>(*(training_data.inputs[i]));
            SeqLabelerOutput *output = static_cast<SeqLabelerOutput>(*(training_data.outputs[i]));
            RawSequence &raw_seq = training_data.raw_sequences_[i];

            feature_extractor.generateCache(input, output, raw_seq);


        }
    }

private:
    GreedyChunker(const GreedyChunker&) = delete;
    GreedyChunker&operator=(const GreedyChunker&) = delete;
};

#endif // SNNOW_GREEDYCHUNKER_H
