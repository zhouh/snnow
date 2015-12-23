/*************************************************************************
	> File Name: FeatureExtractor.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 02:42:44 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREEXTRACTOR_H_
#define _CHUNKER_FEATUREEXTRACTOR_H_

#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <tr1/unordered_map>

#include "FeatureEmbedding.h"

#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Instance.h"
#include "Example.h"
#include "State.h"

class FeatureExtractor{
    std::vector<std::string> m_lKnownWords;
    std::vector<std::string> m_lKnownTags;
    std::vector<std::string> m_lKnownLabels;
    std::tr1::unordered_map<std::string, int> m_mWord2Idx;
    std::tr1::unordered_map<std::string, int> m_mTag2Idx;
    std::tr1::unordered_map<std::string, int> m_mLabel2Idx;

    std::vector<std::string> m_lKnownCapfeats;
    std::tr1::unordered_map<std::string, int> m_mCapfeat2Idx;

    int wordNullIdx;
    int wordUnkIdx;
    int wordNumIdx;
    int tagNullIdx;
    int tagUnkIdx;
    int labelNullIdx;
    int labelUnkIdx;

    int capfeatNullIdx;
    int capfeatUnkIdx;

public:
    static std::string nullstr;
    static std::string unknownstr;
    static std::string numberstr;

    static std::string noncapitalstr;
    static std::string allcapitalstr;
    static std::string firstlettercapstr;
    static std::string hadonecapstr;

public:
    FeatureExtractor() {}

    int size() {
        return m_mWord2Idx.size() + m_mTag2Idx.size() + m_mLabel2Idx.size() + m_mCapfeat2Idx.size();
    }

    void printDict() {
        std::cerr << "knownwords size: " << m_lKnownWords.size() << std::endl;
        std::cerr << "knowntags size: " << m_lKnownTags.size() << std::endl;
        std::cerr << "knownlabels size: " << m_lKnownLabels.size() << std::endl;
    }

    int word2WordIdx(const std::string &s) {
        if (isNumber(s)) {
            return wordNumIdx;
        }

        auto it = m_mWord2Idx.find(s);

        return (it == m_mWord2Idx.end()) ? wordUnkIdx : it->second;
    }

    int tag2TagIdx(const std::string &s) {
        auto it = m_mTag2Idx.find(s);

        return (it == m_mTag2Idx.end()) ? tagUnkIdx : it->second;
    }

    int label2LabelIdx(const std::string &s) {
        auto it = m_mLabel2Idx.find(s);

        if (it == m_mLabel2Idx.end()) {
            std::cerr << "Chunk label not found: " << s << std::endl;
            exit(1);
        }

        return it->second;
    }

    int word2CapfeatIdx(const std::string &s) {
        bool isNoncap = true;
        bool isAllcap = true;
        bool isFirstCap = false;
        bool isHadCap  = false;

        if (isupper(s[0])) {
            isFirstCap = true;
        }

        for (char ch : s) {
            if (isupper(ch)) {
                isHadCap = true;
                isNoncap = false;
            } else {
                isAllcap = false;
            }
        }

        if (isNoncap) {
            return m_mCapfeat2Idx[noncapitalstr];
        }

        if (isAllcap) {
            return m_mCapfeat2Idx[allcapitalstr];
        }

        if (isFirstCap) {
            return m_mCapfeat2Idx[firstlettercapstr];
        }

        if (isHadCap) {
            return m_mCapfeat2Idx[hadonecapstr];
        }

        std::cerr << "word2CapfeatIdx wrong: " << s << std::endl;
        exit(1);
    }

    const std::vector<std::string>& getKnownLabels() const {
        return m_lKnownLabels;
    }

    static std::string processWord(const std::string &word) {
        std::string low_word(word);

        std::transform(low_word.begin(), low_word.end(), low_word.begin(), ::tolower);

        return low_word;
    }

    static bool isNumber(const std::string &word) {
        for (char ch : word) {
            if (!isdigit(ch)){
                return false;
            }
        }

        return true;
    }

    void getDictionaries(const ChunkedDataSet &goldSet);

    void generateInstanceCache(Instance &inst);

    void generateInstanceSetCache(InstanceSet &instSet);

    void extractFeature(State &state, Instance &inst, std::vector<int> &features);

    void generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples);

    int readPretrainEmbeddings(std::string &pretrainFile, FeatureEmbedding &fe);
};

#endif
