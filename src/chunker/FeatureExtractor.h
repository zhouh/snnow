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

    int wordNullIdx;
    int wordUnkIdx;
    int tagNullIdx;
    int tagUnkIdx;
    int labelNullIdx;
    int labelUnkIdx;

public:
    static std::string nullstr;
    static std::string unknownstr;
    const static int featureNum = 1;

public:
    FeatureExtractor() {}

    int size() {
        return m_mWord2Idx.size() + m_mTag2Idx.size() + m_mLabel2Idx.size();
    }

    void printDict() {
        std::cout << "knownwords size: " << m_lKnownWords.size() << std::endl;
        std::cout << "knowntags size: " << m_lKnownTags.size() << std::endl;
        std::cout << "knownlabels size: " << m_lKnownLabels.size() << std::endl;
    }

    int getWordIdx(const std::string &s) {
        auto it = m_mWord2Idx.find(s);

        return (it == m_mWord2Idx.end()) ? wordUnkIdx : it->second;
    }

    int getTagIdx(const std::string &s) {
        auto it = m_mTag2Idx.find(s);

        return (it == m_mTag2Idx.end()) ? tagUnkIdx : it->second;
    }

    int getLabelIdx(const std::string &s) {
        auto it = m_mLabel2Idx.find(s);

        if (it == m_mLabel2Idx.end()) {
            std::cerr << "Chunk label not found: " << s << std::endl;
            exit(1);
        }

        return it->second;
    }

    const std::vector<std::string>& getKnownLabels() const {
        return m_lKnownLabels;
    }

    void getDictionaries(const ChunkedDataSet &goldSet);

    void generateInstanceCache(Instance &inst);

    void generateInstanceSetCache(InstanceSet &instSet);

    void extractFeature(State &state, Instance &inst, std::vector<int> &features);

    void generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples);

    int readPretrainEmbeddings(std::string &pretrainFile, FeatureEmbedding &fe);
};

#endif
