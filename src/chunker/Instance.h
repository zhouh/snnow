/*************************************************************************
	> File Name: Instance.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 09:47:55 PM CST
 ************************************************************************/
#ifndef _CHUNKER_INSTANCE_H_
#define _CHUNKER_INSTANCE_H_

#include <algorithm>
#include <vector>
#include <iostream>

#include "ChunkedSentence.h"

class Instance {
public:
    ChunkerInput input;
    std::vector<int> tagCache;
    std::vector<int> wordCache;

    Instance(ChunkerInput input) {
        this->input = input;
    }

    int size() {
        return this->input.size();
    }

    void print() {
        for (auto &wordTag : input)
            std::cout << wordTag.first << "_" << wordTag.second << " ";
        std::cout << std::endl;
    }

    ~Instance() {}
};

typedef std::vector<Instance> InstanceSet;
#endif 
