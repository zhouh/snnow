/*************************************************************************
	> File Name: DataManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 12:53:45 PM CST
 ************************************************************************/
#ifndef _CHUNKER_DATAMANAGER_H_
#define _CHUNKER_DATAMANAGER_H_

#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>
#include <memory>

#include "Instance.h"
#include "DictManager.h"
#include "ChunkedSentence.h"

class DataManager{
public:
    std::tr1::unordered_map<std::string, std::shared_ptr<DictManager>> m_mStr2DictManager;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    // static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;

public:
    DataManager() {
        m_mStr2DictManager[WORDDESCRIPTION] = std::shared_ptr<DictManager>(new WordDataManager());
        m_mStr2DictManager[POSDESCRIPTION] = std::shared_ptr<DictManager>(new POSDataManager());
        // m_mStr2DictManager[LABELDESCRIPTION] = std::shared_ptr<DictManager>(new LabelDataManager());
        m_mStr2DictManager[CAPDESCRIPTION] = std::shared_ptr<DictManager>(new CapitalDataManager());
    }
    ~DataManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        for (auto &it : m_mStr2DictManager) {
            it.second->makeDictionaries(goldSet);
        }
    }

    void generateInstanceCache(Instance &inst);

private:
    DataManager(const DataManager &dManager)  = delete;
    DataManager& operator= (const DataManager &dManager) = delete;
};

#endif
