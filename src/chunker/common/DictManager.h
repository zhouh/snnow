/*************************************************************************
	> File Name: DictManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 12:53:45 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_DICTMANAGER_H_
#define _CHUNKER_COMMON_DICTMANAGER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>
#include <memory>

#include "Config.h"
#include "Dictionary.h"
#include "LabeledSequence.h"

#define DEBUG

class DictManager{
private:
    std::tr1::unordered_map<std::string, std::shared_ptr<Dictionary>> m_mStr2Dict;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;

public:
    DictManager() { }
    ~DictManager() {}

    void init(const ChunkedDataSet &goldSet);

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        for (auto &it : m_mStr2Dict) {
            it.second->makeDictionaries(goldSet);
        }
    }

    const std::shared_ptr<Dictionary>& getDictionaryOf(const std::string dictName) const {
        return m_mStr2Dict.find(dictName)->second;
    }

    void saveDictManager(std::ostream &os);

    void loadDictManager(std::istream &is);
private:
    DictManager(const DictManager &dManager) = delete;
    DictManager& operator= (const DictManager &dManager) = delete;
};

#undef DEBUG

#endif
