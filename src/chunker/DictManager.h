/*************************************************************************
	> File Name: DictManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 03:24:57 PM CST
 ************************************************************************/
#ifndef _CHUNKER_DICTMANAGER_H_
#define _CHUNKER_DICTMANAGER_H_

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>

#include "ChunkedSentence.h"

#define DEBUG

class DictManager {
public:
    std::vector<std::string> m_lKnownElements;
    std::tr1::unordered_map<std::string, int> m_mElement2Idx;

    int nullIdx;
    int unkIdx;

    static const std::string nullstr;
    static const std::string unknownstr;

public:
    DictManager() {}
    virtual ~DictManager() {}

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

private:
    DictManager(const DictManager &dManager) = delete;
    DictManager& operator= (const DictManager &dManager) = delete;
};

class WordDictManager : public DictManager {
public:
    int numberIdx;

    static const std::string numberstr;

public:
    WordDictManager() {}
    ~WordDictManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);

    int element2Idx(const std::string &s);

    static std::string processWord(const std::string &word);

    static bool isNumber(const std::string &word);
};

class POSDictManager : public DictManager {
public:
    POSDictManager() {}
    ~POSDictManager() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

class LabelDictManager : public DictManager {
public:
    LabelDictManager() { }
    ~LabelDictManager() {}

    int element2Idx(const std::string &s);

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

class CapitalDictManager : public DictManager {
public:
    static const std::string noncapitalstr;
    static const std::string allcapitalstr;
    static const std::string firstlettercapstr;
    static const std::string hadonecapstr;

public:
    CapitalDictManager() {}
    ~CapitalDictManager() {}

    int element2Idx(const std::string &s);

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

#endif
