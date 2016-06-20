/*************************************************************************
	> File Name: SeqLabelerInput.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:37:24 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELERINPUT_H
#define SNNOW_SEQLABELERINPUT_H

#include <vector>
#include <string>
#include <iostream>

#include "base/Input.h"

class SeqLabelerInput : public Input {
public:
    // The following are cached information generated from SequenceInput based on dictionaries
    std::vector<int> word_cache_; // word indexes
    std::vector<int> tag_cache_;  // POS-tag indexes
    std::vector<std::vector<int>> bi_tag_cache_; // bigram POS-tag indexes
    std::vector<std::vector<int>> affix_cache_; // letter-level preffix and suffix with lenght of 1~3 indexes
    std::vector<int> capital_cache_;    // capital information telling if a word is all uppercase, all lowercase, first letter uppercase or at least one letter uppercase

public:
    SeqLabelerInput() = default;

    ~SeqLabelerInput() = default;
};

#endif // SNNOW_SEQLABELERINPUT_H


