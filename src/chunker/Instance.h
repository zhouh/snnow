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
#include <memory>
#include <iostream>

#include "ChunkedSentence.h"
#include "DictManager.h"

class Instance;

typedef std::vector<Instance> InstanceSet;

class Instance {
public:
    ChunkerInput input;
    std::vector<int> tagCache;
    std::vector<int> wordCache;
    std::vector<int> capfeatCache;

    Instance(ChunkerInput input) {
        this->input = input;
    }

    int size() {
        return this->input.size();
    }

    void print() {
        for (auto &wordTag : input)
            std::cerr << wordTag.first << "_" << wordTag.second << " ";
        std::cerr << std::endl;
    }

    ~Instance() {}

    static void generateInstanceCache(DictManager &dictManager, Instance &inst) {
        inst.wordCache.resize(inst.input.size());
        inst.tagCache.resize(inst.input.size());
        inst.capfeatCache.resize(inst.input.size());

        // const std::shared_ptr<DictManager> &wordDict   = dictManager.m_mStr2Dict[DictManager::WORDDESCRIPTION];
        // const std::shared_ptr<DictManager> &posTagDict = dictManager.m_mStr2Dict[DictManager::POSDESCRIPTION];
        // const std::shared_ptr<DictManager> &capDict    = dictManager.m_mStr2Dict[DictManager::CAPDESCRIPTION];
        
        int index = 0;
        for (auto &e : inst.input) {
            dictManager.m_mStr2Dict[DictManager::WORDDESCRIPTION];
            int wordIdx = dictManager.m_mStr2Dict[DictManager::WORDDESCRIPTION]->element2Idx(WordDictionary::processWord(e.first));
            int tagIdx  = dictManager.m_mStr2Dict[DictManager::POSDESCRIPTION]->element2Idx(e.second);
            int capIdx  = dictManager.m_mStr2Dict[DictManager::CAPDESCRIPTION]->element2Idx(e.first);

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

#endif 
