/*************************************************************************
	> File Name: ChunkerDataManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 12:53:45 PM CST
 ************************************************************************/
#ifndef _CHUNKER_CHUNKERDATAMANAGER_H_
#define _CHUNKER_CHUNKERDATAMANAGER_H_

#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>
#include <memory>

#include "Instance.h"
#include "DataManager.h"
#include "ChunkedSentence.h"

class ChunkerDataManager{
public:
    std::tr1::unordered_map<std::string, std::shared_ptr<DataManager>> m_mDesc2dataManager;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    // static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;

public:
    ChunkerDataManager() {
        m_mDesc2dataManager[WORDDESCRIPTION] = std::shared_ptr<DataManager>(new WordDataManager());
        m_mDesc2dataManager[POSDESCRIPTION] = std::shared_ptr<DataManager>(new POSDataManager());
        // m_mDesc2dataManager[LABELDESCRIPTION] = std::shared_ptr<DataManager>(new LabelDataManager());
        m_mDesc2dataManager[CAPDESCRIPTION] = std::shared_ptr<DataManager>(new CapitalDataManager());
    }
    ~ChunkerDataManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        for (auto &it : m_mDesc2dataManager) {
            it.second->makeDictionaries(goldSet);
        }
    }

    void generateInstanceCache(Instance &inst);

};

#endif
