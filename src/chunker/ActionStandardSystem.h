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

#include "State.h"
#include "ChunkedSentence.h"
#include "Instance.h"

class ActionStandardSystem {
public:
    std::vector<std::string> knowLabels;
    int nActNum;

    int nOutside;
    int nInside;
    int nBegin;

    ActionStandardSystem() {}

    ~ActionStandardSystem() {}

    void makeTransition(const std::vector<std::string> &knowLabels);

    int label2ActionType(const std::string &label);

    int labelIdx2ActionIdx(const int actionType, const int labelIdx = 0);

    int actionIdx2actionType(const int actionIdx);

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

public:
    void displayLabel2ActionIdx() {
        std::cerr << "Transition Action Info: " << std::endl;
        for (int i = 0; i < nActNum; i++) {
            std::cerr << i << ": " << knowLabels[i] << std::endl;
        }

        std::cerr << "nInside: " << nInside << std::endl;
        std::cerr << "nOutside: " << nOutside << std::endl;
        std::cerr << "nBegin: " << nBegin << std::endl;
    }
};

#endif
