/*************************************************************************
	> File Name: ChunkerDataManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 02:39:54 PM CST
 ************************************************************************/
#include "ChunkerDataManager.h"

const std::string ChunkerDataManager::WORDDESCRIPTION = "word";
const std::string ChunkerDataManager::POSDESCRIPTION = "pos";
// const std::string ChunkerDataManager::LABELDESCRIPTION = "label";
const std::string ChunkerDataManager::CAPDESCRIPTION  = "capital";

void ChunkerDataManager::generateInstanceCache(Instance &inst) {
    inst.wordCache.resize(inst.input.size());
    inst.tagCache.resize(inst.input.size());
    inst.capfeatCache.resize(inst.input.size());

    int index = 0;
    for (auto &e : inst.input) {
        int wordIdx = m_mDesc2dataManager[ChunkerDataManager::WORDDESCRIPTION]->element2Idx(WordDataManager::processWord(e.first));
        int tagIdx = m_mDesc2dataManager[ChunkerDataManager::POSDESCRIPTION]->element2Idx(e.second);
        int capfeatIdx = m_mDesc2dataManager[ChunkerDataManager::CAPDESCRIPTION]->element2Idx(e.first);

        inst.wordCache[index] = wordIdx;
        inst.tagCache[index] = tagIdx;
        inst.capfeatCache[index] = capfeatIdx;

        index++;
    }
}
