/*************************************************************************
	> File Name: DictManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 12:53:45 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_DICTMANAGER_H_
#define _CHUNKER_COMMON_DICTMANAGER_H_

#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>
#include <memory>

#include "Dictionary.h"
#include "LabeledSequence.h"

class DictManager{
private:
    std::tr1::unordered_map<std::string, std::shared_ptr<Dictionary>> m_mStr2Dict;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    // static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;

public:
    DictManager() { }
    ~DictManager() {}

    void init(const ChunkedDataSet &goldSet) {
        m_mStr2Dict[WORDDESCRIPTION] = std::shared_ptr<Dictionary>(new WordDictionary());
        m_mStr2Dict[POSDESCRIPTION] = std::shared_ptr<Dictionary>(new POSDictionary());
        // m_mStr2Dict[LABELDESCRIPTION] = std::shared_ptr<Dictionary>(new LabelDictionary());
        m_mStr2Dict[CAPDESCRIPTION] = std::shared_ptr<Dictionary>(new CapitalDictionary());

        makeDictionaries(goldSet);
    }

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        for (auto &it : m_mStr2Dict) {
            it.second->makeDictionaries(goldSet);
        }
    }

    const std::shared_ptr<Dictionary>& getDictionaryOf(const std::string dictName) const {
        return m_mStr2Dict.find(dictName)->second;
    }

private:
    DictManager(const DictManager &dManager) = delete;
    DictManager& operator= (const DictManager &dManager) = delete;
};

#endif
