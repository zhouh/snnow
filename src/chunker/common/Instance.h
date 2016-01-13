/*************************************************************************
	> File Name: Instance.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 09:47:55 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_INSTANCE_H_
#define _CHUNKER_COMMON_INSTANCE_H_

#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>

#include "LabeledSequence.h"
#include "DictManager.h"

class Instance;

typedef std::vector<Instance> InstanceSet;

class Instance {
public:
    SequenceInput input;
    std::vector<int> tagCache;
    std::vector<int> wordCache;
    std::vector<int> capfeatCache;

    Instance(SequenceInput input) {
        this->input = input;
    }

    int size() const {
        return this->input.size();
    }

    ~Instance() {}

    static void generateInstanceCache(DictManager &dictManager, Instance &inst) {
        inst.wordCache.resize(inst.input.size());
        inst.tagCache.resize(inst.input.size());
        inst.capfeatCache.resize(inst.input.size());

        const std::shared_ptr<Dictionary> &wordDict   = dictManager.getDictionaryOf(DictManager::WORDDESCRIPTION);
        const std::shared_ptr<Dictionary> &posTagDict = dictManager.getDictionaryOf(DictManager::POSDESCRIPTION);
        const std::shared_ptr<Dictionary> &capDict    = dictManager.getDictionaryOf(DictManager::CAPDESCRIPTION);
        
        int index = 0;
        for (auto &e : inst.input) {
            int wordIdx = wordDict->element2Idx(WordDictionary::processWord(e.first));
            int tagIdx  = posTagDict->element2Idx(e.second);
            int capIdx  = capDict->element2Idx(e.first);

            inst.wordCache[index]    = wordIdx;
            inst.tagCache[index]     = tagIdx;
            inst.capfeatCache[index] = capIdx;

            index++;
        }
    }

    static void generateInstanceSetCache(DictManager &dictManager, InstanceSet &instSet) {
        for (Instance &inst : instSet) {
            generateInstanceCache(dictManager, inst);
        }
    }
};

inline std::ostream& operator << (std::ostream& os, const Instance &inst) {
    for (int i = 0; i < inst.size(); i++) {
        os << i << ":\t";
        os << inst.input[i].first << "\t" << inst.input[i].second << "\t";
        os << inst.wordCache[i] << "\t" << inst.tagCache[i] << "\t" << inst.capfeatCache[i];
        os << std::endl;
    }

    return os;
}

#endif 
