/*************************************************************************
	> File Name: ActionStandardSystem.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 24 Nov 2015 02:31:25 PM CST
 ************************************************************************/
#ifndef _CHUNKER_ACTIONSTANDARDSYSTEM_H_
#define _CHUNKER_ACTIONSTANDARDSYSTEM_H_

#include <string>
#include <vector>
#include <iostream>
#include <tr1/unordered_map>
#include <unordered_set>

#include "State.h"
#include "ChunkedSentence.h"
#include "Instance.h"

class LabelManager {
public:
    std::vector<std::string> m_lKnownLabels;
    std::tr1::unordered_map<std::string, int> m_mLabel2Idx;

public:
    LabelManager() {}
    ~LabelManager() {}

    int size() {
        return static_cast<int>(m_mLabel2Idx.size());
    }

    const std::vector<std::string>& getKnownLabels() const {
        return m_lKnownLabels;
    }

    int label2Idx(const std::string &s) {
        auto it = m_mLabel2Idx.find(s);

        if (it == m_mLabel2Idx.end()) {
            std::cerr << "chunk label not found: " << s << std::endl;
            exit(0);
        }

        return it->second;
    }

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        using std::unordered_set;
        using std::string;

        unordered_set<string> labelSet;

        for (auto &sent: goldSet) {
            for (auto &cw : sent.getChunkedWords()) {
                labelSet.insert(cw.label);
            }
        }
#ifdef DEBUG
        std::cerr << "  labelSet size: " << labelSet.size() << std::endl;
#endif
        int idx = 0;

        for (auto &l : labelSet) {
            m_mLabel2Idx[l] = idx++, m_lKnownLabels.push_back(l);
        }
    }

    void printDict() {
        std::cerr << "known feature size: " << m_lKnownLabels.size() << std::endl;
    }
};

class ActionStandardSystem {
public:
    LabelManager labelManager;
    std::vector<std::string> knowLabels;
    int nActNum;

    int nOutside;
    int nInside;
    int nBegin;

    ActionStandardSystem() {}

    ~ActionStandardSystem() {}

    void init(const ChunkedDataSet &goldSet);

    int label2ActionType(const std::string &label);

    int labelIdx2ActionIdx(const int actionType, const int labelIdx = 0);

    int actionIdx2ActionType(const int actionIdx);

    int actionIdx2LabelIdx(const int actionIdx);

    void move(const State &srcState, State &dstState, const CScoredTransition &transition);

    int standardMove(State &state, ChunkedSentence &gSent, std::vector<int> labelIndexesCache);

    void generateValidActs(State &state, std::vector<int> &validActs);

    void generateOutput(const State &state, ChunkedSentence &sent);

    int getOutsideIndex() {
        return nOutside;
    }

    int getInsideIndex() {
        return nInside;
    }

    int getBeginIndex() {
        return nBegin;
    }

private:
    void doOutsideMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void doInsideMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void doBeginMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void makeTransition(const std::vector<std::string> &knowLabels);

public:
    void displayLabel2ActionIdx() {
        std::cerr << "\tTransition Action Info: " << std::endl;
        for (int i = 0; i < nActNum; i++) {
            std::cerr << "\t\t" << i << ": " << knowLabels[i] << std::endl;
        }

        std::cerr << "\t\tnInside: " << nInside << std::endl;
        std::cerr << "\t\tnOutside: " << nOutside << std::endl;
        std::cerr << "\t\tnBegin: " << nBegin << std::endl;
    }
};

#endif
