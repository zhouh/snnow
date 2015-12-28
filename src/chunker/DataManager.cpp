/*************************************************************************
	> File Name: DataManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 02:39:54 PM CST
 ************************************************************************/
#include "DataManager.h"

const std::string DataManager::WORDDESCRIPTION = "word";
const std::string DataManager::POSDESCRIPTION = "pos";
// const std::string DataManager::LABELDESCRIPTION = "label";
const std::string DataManager::CAPDESCRIPTION  = "capital";

void DataManager::generateInstanceCache(Instance &inst) {
    inst.wordCache.resize(inst.input.size());
    inst.tagCache.resize(inst.input.size());
    inst.capfeatCache.resize(inst.input.size());

    int index = 0;
    for (auto &e : inst.input) {
        int wordIdx = m_mStr2DictManager[DataManager::WORDDESCRIPTION]->element2Idx(WordDataManager::processWord(e.first));
        int tagIdx = m_mStr2DictManager[DataManager::POSDESCRIPTION]->element2Idx(e.second);
        int capfeatIdx = m_mStr2DictManager[DataManager::CAPDESCRIPTION]->element2Idx(e.first);

        inst.wordCache[index] = wordIdx;
        inst.tagCache[index] = tagIdx;
        inst.capfeatCache[index] = capfeatIdx;

        index++;
    }
}
