/*************************************************************************
	> File Name: SeqLabelerDataset.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:35:50 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELERDATASET_H
#define SNNOW_SEQLABELERDATASET_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "base/Input.h"
#include "base/Output.h"
#include "DataSet.h"
#include "SeqLabelerInput.h"
#include "SeqLabelerOutput.h"

// The raw text sequence of a sentence, in chunker, it consists of vector of (word, POS-tag, label)
typedef std::vector<std::vector<std::string>> RawSequence;
extern  std::ostream& operator<< (std::ostream &os, RawSequence &raw_sequence);

extern  std::istream& operator>> (std::istream &is, RawSequence &raw_sequence);

extern RawSequence transformEtypeFormat2BIOFormat(RawSequence &raw_seq);

class SeqLabelerDataSet : public DataSet {
public:
    std::vector<RawSequence> raw_sequences_;

public:
    SeqLabelerDataSet(std::string file_name) {
        std::ifstream is(file_name.c_str());

        size = 0;

        while(true) {
            if (!is) {
                break;
            }

            RawSequence raw_seq;

            is >> raw_seq;

            raw_sequences_.push_back(raw_seq);

            inputs.push_back(static_cast<Input*>(new SeqLabelerInput));
            outputs.push_back(static_cast<Output*>(new SeqLabelerOutput));

            size++;
        }
    }

    ~SeqLabelerDataSet() {
        for (auto i = 0; i < inputs.size(); i++) {
            delete inputs[i];
            delete outputs[i];
        }
    }

    friend std::ostream& operator<< (std::ostream &os, SeqLabelerDataSet &seq_dataset);

private:
    SeqLabelerDataSet(const SeqLabelerDataSet&) = delete;
    SeqLabelerDataSet& operator= (const SeqLabelerDataSet&) = delete;
};

std::ostream& operator<< (std::ostream &os, SeqLabelerDataSet &seq_dataset);

#endif // SNNOW_SEQLABELERDATASET_H
