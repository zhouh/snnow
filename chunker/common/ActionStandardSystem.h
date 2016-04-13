/*************************************************************************
	> File Name: ActionStandardSystem.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 24 Nov 2015 02:31:25 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_ACTIONSTANDARDSYSTEM_H_
#define _CHUNKER_COMMON_ACTIONSTANDARDSYSTEM_H_

#include <string>
#include <vector>
#include <iostream>

#include "Dictionary.h"
#include "State.h"
#include "LabeledSequence.h"
#include "Instance.h"
#include "Dictionary.h"

class LabelManager {
private:
    LabelDictionary labelDict;

public:
    LabelManager() {}
    ~LabelManager() {}

    int size() {
        return labelDict.size();
    }

    const std::vector<std::string> getKnownLabels() const {
        std::vector<std::string> knownLabels;

        for (const std::string &s : labelDict.getKnownElements()) {
            if (s != LabelDictionary::nullstr && s != LabelDictionary::unknownstr) {
                knownLabels.push_back(s);
            }
        }
        return knownLabels;
    }

    int label2Idx(const std::string &s) const;

    void makeDictionaries(const ChunkedDataSet &goldSet);

    void saveLabelManager(std::ostream &os) {
        labelDict.saveDictionary(os);
    }

    void loadLabelManager(std::istream &is) {
        labelDict.loadDictionary(is);
    }
private:
    LabelManager(const LabelManager &lm) = delete;
    LabelManager& operator= (const LabelManager &lm) = delete;
};

class ActionStandardSystem {
private:
    LabelManager labelManager;
    std::vector<std::string> knowLabels;
    int nActNum;

    int nOutside;
    int nInside;
    int nBegin;

public:
    ActionStandardSystem() {}

    ~ActionStandardSystem() {}

    void init(const ChunkedDataSet &goldSet);

    int label2ActionType(const std::string &label);

    int labelIdx2ActionIdx(const int actionType, const int labelIdx = 0);

    int actionIdx2ActionType(const int actionIdx);

    int actionIdx2LabelIdx(const int actionIdx);

    void move(const State &srcState, State &dstState, const CScoredTransition &transition);

    int standardMove(State &state, LabeledSequence &gSent, std::vector<int> labelIndexesCache);

    void generateValidActs(State &state, std::vector<int> &validActs);

    void generateOutput(const State &state, LabeledSequence &sent);

    int getActNumber() {
        return nActNum;
    }

    const LabelManager& getLabelManager() const {
        return labelManager;
    }

    int getOutsideIndex() {
        return nOutside;
    }

    int getInsideIndex() {
        return nInside;
    }

    int getBeginIndex() {
        return nBegin;
    }

    void saveActionSystem(std::ostream &os);

    void loadActionSystem(std::istream &is);

private:
    void doOutsideMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void doInsideMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void doBeginMove(State &srcState, State &dstState, const CScoredTransition &transition);

    void makeTransition(const std::vector<std::string> &knowLabels);

private:
    ActionStandardSystem(const ActionStandardSystem &transSystem) = delete;
    ActionStandardSystem& operator= (const ActionStandardSystem &transSystem) = delete;

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
