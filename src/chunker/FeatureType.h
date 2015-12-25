/*************************************************************************
	> File Name: FeatureType.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 03:24:57 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATURETYPE_H_
#define _CHUNKER_FEATURETYPE_H_

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>

#include "ChunkedSentence.h"

#define DEBUG

class FeatureType {
public:
    std::vector<std::string> m_lKnownFeatures;
    std::tr1::unordered_map<std::string, int> m_mFeat2Idx;

    int nullIdx;
    int unkIdx;

    static const std::string nullstr;
    static const std::string unknownstr;

public:
    FeatureType() {}
    virtual ~FeatureType() {}

    int size() {
        return static_cast<int>(m_mFeat2Idx.size());
    }

    const std::vector<std::string>& getKnownFeatures() const {
        return m_lKnownFeatures;
    }

    virtual int feat2FeatIdx(const std::string &s) {
        auto it = m_mFeat2Idx.find(s);

        return (it == m_mFeat2Idx.end()) ? unkIdx: it->second;
    }

    virtual void getDictionaries(const ChunkedDataSet &goldSet) = 0;

    void printDict() {
        std::cerr << "known feature size: " << m_lKnownFeatures.size() << std::endl;
    }
};

class WordFeature : public FeatureType {
public:
    int numberIdx;

    static const std::string numberstr;

public:
    WordFeature() {}
    ~WordFeature() {}

    void getDictionaries(const ChunkedDataSet &goldSet);

    int feat2FeatIdx(const std::string &s);

    static std::string processWord(const std::string &word);

    static bool isNumber(const std::string &word);
};

class POSFeature : public FeatureType {
public:
    POSFeature() {}
    ~POSFeature() {}

    void getDictionaries(const ChunkedDataSet &goldSet);
};

class LabelFeature : public FeatureType {
public:
    LabelFeature() { }
    ~LabelFeature() {}

    int feat2FeatIdx(const std::string &s);

    void getDictionaries(const ChunkedDataSet &goldSet);
};

class CapitalFeature : public FeatureType {
public:
    static const std::string noncapitalstr;
    static const std::string allcapitalstr;
    static const std::string firstlettercapstr;
    static const std::string hadonecapstr;

public:
    CapitalFeature() {}
    ~CapitalFeature() {}

    int feat2FeatIdx(const std::string &s);

    void getDictionaries(const ChunkedDataSet &goldSet);
};

#endif
