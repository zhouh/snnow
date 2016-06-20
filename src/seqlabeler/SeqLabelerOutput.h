/*************************************************************************
	> File Name: SeqLabelerOutput.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 07:06:54 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELEROUTPUT_H
#define SNNOW_SEQLABELEROUTPUT_H

#include <string>
#include <vector>

class SeqLabelerOutput : public Output {
public:
    std::vector<int> output_label_;

public:
    SeqLabelerOutput() = default;

    ~SeqLabelerOutput() = default;

    int outputAt(const int index) {
        return output_label_[index];
    }

    int sequenceLength() {
        return output_label_.size();
    }
};

#endif // SNNOW_SEQLABELEROUTPUT_H
