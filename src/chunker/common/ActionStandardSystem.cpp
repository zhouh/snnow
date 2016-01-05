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

int LabelManager::label2Idx(const std::string &s) const {
    auto it = m_mLabel2Idx.find(s);

    if (it == m_mLabel2Idx.end()) {
        std::cerr << "chunk label not found: " << s << std::endl;
        exit(0);
    }

    return it->second;
}

void LabelManager::makeDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> labelSet;

    for (auto &sent: goldSet) {
        for (auto &cw : sent.getLabeledTerms()) {
            labelSet.insert(cw.label);
        }
    }

    std::cerr << "  labelSet size: " << labelSet.size() << std::endl;

    int idx = 0;

    for (auto &l : labelSet) {
        m_mLabel2Idx[l] = idx++, m_lKnownLabels.push_back(l);
    }
}

void ActionStandardSystem::init(const ChunkedDataSet &goldSet) {
    labelManager.makeDictionaries(goldSet);
    makeTransition(labelManager.getKnownLabels());
}

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

int ActionStandardSystem::actionIdx2ActionType(const int actionIdx) {
    if (actionIdx == nOutside || actionIdx == nInside) {
        return actionIdx;
    } else {
        return nBegin;
    }
}

int ActionStandardSystem::actionIdx2LabelIdx(const int actionIdx) {
    const int actionType = actionIdx2ActionType(actionIdx);

    if (actionType == nOutside || actionType == nInside) {
        return actionType;
    } else {
        return actionIdx;
    }
}

void ActionStandardSystem::move(const State &srcState, State &dstState, const CScoredTransition &transition) {
    int actionType = actionIdx2ActionType(transition.action);

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

int ActionStandardSystem::standardMove(State &state, LabeledSequence &gSent, std::vector<int> labelIndexesCache) {
    if (state.complete()) {
        std::cerr << "The chunking state is completed!" << std::endl;
        exit(1);
    }

    int index = state.m_nIndex;

    int actionType = label2ActionType(gSent.getLabeledTerms()[ index+ 1].label);
    int labelIdx = labelIndexesCache[index + 1];

    return labelIdx2ActionIdx(actionType, labelIdx);
}

void ActionStandardSystem::generateValidActs(State &state, std::vector<int> &validActs) {
    validActs.resize(nActNum, 0);

    if (state.m_nIndex == -1 || state.last_action == nOutside) {
       validActs[nInside] = -1;
    }
}

void ActionStandardSystem::generateOutput(const State &state, LabeledSequence &sent) {
    const State *ptr = &state;
    while (ptr->previous_ != nullptr) {
#ifdef DEBUGX
        std::cerr << "current m_nIndex: " << ptr->m_nIndex << "\tlabel Idx: " << actionIdx2LabelIdx(ptr->last_action) << "\t"<< sent.m_lLabeledTerms[ptr->m_nIndex].word << "\t" << sent.m_lLabeledTerms[ptr->m_nIndex].tag << std::endl;
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
