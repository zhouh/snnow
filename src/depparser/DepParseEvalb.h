//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DEPPARSEEVALB_H
#define SNNOW_DEPPARSEEVALB_H

#include "base/Evalb.h"
#include "DepParseTree.h"

class DepParseEvalb : public Evalb {

    double evalb(std::vector<Input> &inputs,
                 std::vector<Output> &predicted_outputs,
                 std::vector<Output> &gold_outputs) {

        predicted_outputs = static_cast<std::vector<DepParseTree>>(predicted_outputs);
        gold_outputs = static_cast<std::vector<DepParseInput>>(predicted_outputs);
        assert(predicted_outputs.size() == gold_outputs.size());

        double UAS; // return value

        int correct_head_num = 0;
        int correct_arc_num = 0;
        int arc_sum = 0;

        for (int i = 0; i < predicted_outputs.size(); i++) {

            assert(predicted_outputs[i].size == gold_outputs[i].size);

            for (int j = 1; j < predicts[i].size; j++) {

                std::string tag = predicts[i].nodes[j].tag;
                if (puncs.find(tag) != puncs.end())
                    continue;

                sumArc++;
                if (predicts[i].nodes[j].head == golds[i].nodes[j].head) {
                    correctHeadNum++;
                    if (predicts[i].nodes[j].label == golds[i].nodes[j].label)
                        correctArcNum++;
                }
            }
        }

        UAS = (double) correctHeadNum / sumArc; //UAS
//        retval.second = (double) correctArcNum / sumArc;

        return UAS;
    }

    /**
     * will offer different language support
     */
    bool isPunc(std::string punctuation_candidate) {
        std::unordered_set<std::string> puncs = {"``", "''", ".", ",", ":"}; // only for
    }

};


#endif //SNNOW_DEPPARSEEVALB_H
