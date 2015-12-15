/*************************************************************************
	> File Name: ActionStandardSystem.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 24 Nov 2015 04:13:05 PM CST
 ************************************************************************/
#include "ActionStandardSystem.h"

#define DEBUG

#ifdef DEBUG
//#define DEBUG1
#define DEBUG2
#endif

void ActionStandardSystem::makeTransition(const std::vector<std::string> &knowLabels) {
    this->knowLabels = knowLabels;

    for (int i = 0; i < knowLabels.size(); i++) {
        if (knowLabels[i] == "I") {
            nInside = i;
        } else if (knowLabels[i] == "O") {
            nOutside = i;
        } else {
            nBegin = i;
        }
    }

    nActNum = knowLabels.size();
}

int ActionStandardSystem::label2ActionType(const std::string &label) {
    if (label == "O") {
        return nOutside;
    }

    if (label == "I") {
        return nInside;
    }

    return nBegin;
}

int ActionStandardSystem::labelIdx2ActionIdx(const int actionType, const int labelIdx) {
    if (actionType == nOutside || actionType == nInside) {
        return actionType;
    }

    return labelIdx;
}

int ActionStandardSystem::actionIdx2actionType(const int actionIdx) {
    if (actionIdx == nOutside || actionIdx == nInside) {
        return actionIdx;
    } else {
        return nBegin;
    }
}

int ActionStandardSystem::actionIdx2LabelIdx(const int actionIdx) {
    const int actionType = actionIdx2actionType(actionIdx);

    if (actionType == nOutside || actionType == nInside) {
        return actionType;
    } else {
        return actionIdx;
    }
}

void ActionStandardSystem::move(const State &srcState, State &dstState, const CScoredTransition &transition) {
    int actionType = actionIdx2actionType(transition.action);

    if (actionType == nOutside) {
        doOutsideMove(const_cast<State &>(srcState), dstState, transition);
    } else if (actionType == nInside) {
        doInsideMove(const_cast<State &>(srcState), dstState, transition);
    } else if (actionType == nBegin) {
        doBeginMove(const_cast<State &>(srcState), dstState, transition);
    } else {
        std::cerr << "Invalid Move Action Type: " << transition.action << std::endl;
        exit(1);
    }
}

int ActionStandardSystem::standardMove(State &state, ChunkedSentence &gSent, std::vector<int> labelIndexesCache) {
    if (state.complete()) {
        std::cerr << "The chunking state is completed!" << std::endl;
        exit(1);
    }

    int index = state.m_nIndex;

    int actionType = label2ActionType(gSent.getChunkedWords()[ index+ 1].label);
    int labelIdx = labelIndexesCache[index + 1];

    return labelIdx2ActionIdx(actionType, labelIdx);
}

void ActionStandardSystem::generateValidActs(State &state, std::vector<int> &validActs) {
    validActs.resize(nActNum, 0);

    if (state.m_nIndex == -1 || state.last_action == nOutside) {
        validActs[nInside] = -1;
    }
}

void ActionStandardSystem::generateOutput(const State &state, ChunkedSentence &sent) {
    const State *ptr = &state;
    while (ptr->previous_ != nullptr) {
#ifdef DEBUGX
        std::cout << "current m_nIndex: " << ptr->m_nIndex << "\tlabel Idx: " << actionIdx2LabelIdx(ptr->last_action) << "\t"<< sent.m_lChunkedWords[ptr->m_nIndex].word << "\t" << sent.m_lChunkedWords[ptr->m_nIndex].tag << std::endl;
#endif
        sent.setLabel(ptr->m_nIndex, knowLabels[actionIdx2LabelIdx(ptr->last_action)]);

        ptr = ptr->previous_;
    }
}

void ActionStandardSystem::doOutsideMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.m_nIndex = srcState.m_nIndex + 1;
    dstState.previous_ = &srcState;
    dstState.last_action = const_cast<CScoredTransition &>(transition).action;
    dstState.m_nLen = srcState.m_nLen;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}

void ActionStandardSystem::doInsideMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.m_nIndex = srcState.m_nIndex + 1;
    dstState.previous_ = &srcState;
    dstState.last_action = const_cast<CScoredTransition &>(transition).action;
    dstState.m_nLen = srcState.m_nLen;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}

void ActionStandardSystem::doBeginMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.m_nIndex = srcState.m_nIndex + 1;
    dstState.previous_ = &srcState;
    dstState.last_action = const_cast<CScoredTransition &>(transition).action;
    dstState.m_nLen = srcState.m_nLen;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}
