/*************************************************************************
	> File Name: Dictionary.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 03:24:57 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_DICTIONARY_H_
#define _CHUNKER_COMMON_DICTIONARY_H_

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>

#include "LabeledSequence.h"

class Dictionary {
protected:
    std::vector<std::string> m_lKnownElements;
    std::tr1::unordered_map<std::string, int> m_mElement2Idx;

public:
    int nullIdx;
    int unkIdx;

    static const std::string nullstr;
    static const std::string unknownstr;

public:
    Dictionary() {}
    virtual ~Dictionary() {}

    int size() {
        return static_cast<int>(m_mElement2Idx.size());
    }

    const std::vector<std::string>& getKnownElements() const {
        return m_lKnownElements;
    }

    const std::tr1::unordered_map<std::string, int>& getWord2IdxMap() const {
        return m_mElement2Idx;
    }

    virtual int element2Idx(const std::string &s) const {
        auto it = m_mElement2Idx.find(s);

        return (it == m_mElement2Idx.end()) ? unkIdx: it->second;
    }

    virtual void makeDictionaries(const ChunkedDataSet &goldSet) = 0;

    virtual void saveDictionary(std::ostream &os) {
        os << "elementSize" << " " << m_lKnownElements.size() << std::endl;

        for (std::string &e : m_lKnownElements) {
            os << e << " " << m_mElement2Idx[e] << std::endl;
        }
    }

    virtual void loadDictionary(std::istream &is) {
        std::string line;
        std::string tmp;
        int size;
    
        getline(is, line);
        std::istringstream iss(line);
        iss >> tmp >> size;
    
        std::string element;
        int idx;
        for (int i = 0; i < size; i++) {
            getline(is, line);
            std::istringstream iss_j(line);
            iss_j >> element >> idx;

            processElementAndIdx(element, idx);
    
            m_mElement2Idx[element] = idx;
            m_lKnownElements.push_back(element);
        }
    }

    void printDict() {
        for (auto &s : m_lKnownElements) {
            std::cerr << "  " <<  s << ": " << m_mElement2Idx[s] << std::endl;
        }
    }

private:
    Dictionary(const Dictionary &dManager) = delete;
    Dictionary& operator= (const Dictionary &dManager) = delete;

    virtual void processElementAndIdx(const std::string &element, const int idx) {
        if (element == nullstr) {
            nullIdx = idx;
        } else if (element == unknownstr) {
            unkIdx = idx;
        }
    }
};

class WordDictionary : public Dictionary {
private:
    static const std::string numberstr;

public:
    WordDictionary() {}
    ~WordDictionary() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);

    int element2Idx(const std::string &s) const;

    static std::string processWord(const std::string &word);

private:
    static std::string replaceNumber(const std::string &word);
};

class POSDictionary : public Dictionary {
public:
    POSDictionary() {}
    ~POSDictionary() {}

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

class LabelDictionary : public Dictionary {
public:
    LabelDictionary() {}
    ~LabelDictionary() {}

    int element2Idx(const std::string &s) const;

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

// class CurrentLabelDictionary : public Dictionary {
// public:
//     CurrentLabelDictionary() {}
//     ~CurrentLabelDictionary() {}
// 
//     int element2Idx(const std::string &s) const;
// 
//     void makeDictionaries(const ChunkedDataSet &goldSet);
// };

class CapitalDictionary : public Dictionary {
public:
    static const std::string noncapitalstr;
    static const std::string allcapitalstr;
    static const std::string firstlettercapstr;
    static const std::string hadonecapstr;

public:
    CapitalDictionary() {}
    ~CapitalDictionary() {}

    int element2Idx(const std::string &s) const;

    void makeDictionaries(const ChunkedDataSet &goldSet);
};

#endif
