//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DEPPARSEEVALB_H
#define SNNOW_DEPPARSEEVALB_H

#include "base/Evalb.h"
#include "DepParseTree.h"

class DepParseEvalb{

public:
    double evalb(std::vector<DepParseTree> &predicted_outputs,
                 std::vector<DepParseTree> &gold_outputs) {


        assert(predicted_outputs.size() == gold_outputs.size());

        double UAS; // return value

        int correct_head_num = 0;
        int correct_arc_num = 0;
        int arc_sum = 0;

        for (int i = 0; i < predicted_outputs.size(); i++) {

            DepParseTree & predict = predicted_outputs[i];
            DepParseTree & gold_output = gold_outputs[i];

            for (int j = 1; j < predict.size; j++) {

                std::string tag = predict.nodes[j].tag;
                if (isPunc(tag))
                    continue;

                arc_sum++;

                if (predict.nodes[j].head == gold_output.nodes[j].head) {
                    correct_head_num++;
                    if (predict.nodes[j].label == gold_output.nodes[j].label)
                        correct_arc_num++;
                }
            }
        }

        UAS = (double) correct_head_num / arc_sum; //UAS
//        retval.second = (double) correctArcNum / sumArc;

        return UAS;
    }

    /**
     * will offer different language support
     */
    bool isPunc(std::string & punctuation_candidate) {
        static std::unordered_set<std::string> puncs = {"``", "''", ".", ",", ":"}; // only for

        return puncs.find(punctuation_candidate) != puncs.end();
    }

};


#endif //SNNOW_DEPPARSEEVALB_H
