/*************************************************************************
	> File Name: DataManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 03:24:57 PM CST
 ************************************************************************/
#ifndef _CHUNKER_DATAMANAGER_H_
#define _CHUNKER_DATAMANAGER_H_

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>

#include "ChunkedSentence.h"

#define DEBUG

class DataManager {
public:
    std::vector<std::string> m_lKnownElements;
    std::tr1::unordered_map<std::string, int> m_mElement2Idx;

    int nullIdx;
    int unkIdx;

    static const std::string nullstr;
    static const std::string unknownstr;

public:
    DataManager() {}
    virtual ~DataManager() {}

    int size() {
        return static_cast<int>(m_mElement2Idx.size());
    }

    const std::vector<std::string>& getKnownElements() const {
        return m_lKnownElements;
    }

    virtual int element2Idx(const std::string &s) {
        auto it = m_mElement2Idx.find(s);

        return (it == m_mElement2Idx.end()) ? unkIdx: it->second;
    }

    virtual void makeDictionaries(const ChunkedDataSet &goldSet) = 0;

    void printDict() {
        std::cerr << "known feature size: " << m_lKnownElements.size() << std::endl;
    }
};

class WordDataManager : public DataManager {
public:
    int numberIdx;

    static const std::string numberstr;

public:
    WordDataManager() {}
    ~WordDataManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);

    int element2Idx(const std::string &s);

    static std::string processWord(const std::string &word);

    static bool isNumber(const std::string &word);
};

class POSDataManager : public DataManager {
public:
    POSDataManager() {}
    ~POSDataManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

class LabelDataManager : public DataManager {
public:
    LabelDataManager() { }
    ~LabelDataManager() {}

    int element2Idx(const std::string &s);

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

class CapitalDataManager : public DataManager {
public:
    static const std::string noncapitalstr;
    static const std::string allcapitalstr;
    static const std::string firstlettercapstr;
    static const std::string hadonecapstr;

public:
    CapitalDataManager() {}
    ~CapitalDataManager() {}

    int element2Idx(const std::string &s);

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

#endif
