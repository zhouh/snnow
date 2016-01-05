/*************************************************************************
	> File Name: Evalb.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 01 Dec 2015 03:44:19 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_EVALB_H_
#define _CHUNKER_COMMON_EVALB_H_

#include <iostream>
#include <string>
#include <assert.h>

#include "LabeledSequence.h"

class Evalb {
public:
    // return (precision, recall, FB1)
    static std::tuple<double, double, double> eval(ChunkedDataSet &predicts, ChunkedDataSet &golds, bool isEvalNP = false);

    // return: (correct_count, gold_count, predict_count)
    static std::tuple<int, int, int> eval(LabeledSequence &predict, LabeledSequence &gold, bool isEvalNP = false);

private:
    // startOfChunk: checks if a chunk started between the previous and current word
    // arguments:    previous and current chunk tags, previous and current types
    // note:         this code is not capable of handling other chunk representations
    //               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    //               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    static bool startOfChunk(std::string &prevTag, std::string &tag, std::string &prevType, std::string &type);

    // endOfChunk: checks if a chunk ended between the preivous and current word
    // arguments:  previous and current chunk tags, previous and current types
    // note:       this code is not capable of handling other chunk representations
    //             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    //             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    static bool endOfChunk(std::string &prevTag, std::string &tag, std::string prevType, std::string &type);
};

#endif
