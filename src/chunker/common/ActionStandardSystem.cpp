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
    return labelDict.element2Idx(s);
}

void LabelManager::makeDictionaries(const ChunkedDataSet &goldSet) {
    labelDict.makeDictionaries(goldSet);
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

    int index = state.index;

    int actionType = label2ActionType(gSent.getLabeledTerms()[ index+ 1].label);
    int labelIdx = labelIndexesCache[index + 1];

    return labelIdx2ActionIdx(actionType, labelIdx);
}

void ActionStandardSystem::generateValidActs(State &state, std::vector<int> &validActs) {
    validActs.resize(nActNum, 0);

    if (state.index == -1 || state.lastAction == nOutside) {
       validActs[nInside] = -1;
    }
}

void ActionStandardSystem::generateOutput(const State &state, LabeledSequence &sent) {
    const State *ptr = &state;
    while (ptr->prevStatePtr != nullptr) {
#ifdef DEBUGX
        std::cerr << "current index: " << ptr->index << "\tlabel Idx: " << actionIdx2LabelIdx(ptr->lastAction) << "\t"<< sent.m_lLabeledTerms[ptr->index].word << "\t" << sent.m_lLabeledTerms[ptr->index].tag << std::endl;
#endif
        sent.setLabel(ptr->index, knowLabels[actionIdx2LabelIdx(ptr->lastAction)]);

        ptr = ptr->prevStatePtr;
    }
}

void ActionStandardSystem::doOutsideMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.index = srcState.index + 1;
    dstState.prevStatePtr = &srcState;
    dstState.chunkedLabelIds.push_back(actionIdx2LabelIdx(transition.action));
    if (transition.action == nBegin || transition.action == nOutside) {
        if (dstState.onGoChunkIdx == -1) {
            dstState.onGoChunkIdx = 0;
        } else {
            dstState.prevChunkIdx = dstState.currChunkIdx;
            dstState.currChunkIdx = dstState.onGoChunkIdx;
            dstState.onGoChunkIdx = dstState.index;
        }
        dstState.prevChunkIdx = dstState.currChunkIdx;
        dstState.currChunkIdx = dstState.index;
    }
    dstState.lastAction = const_cast<CScoredTransition &>(transition).action;
    dstState.sentLength = srcState.sentLength;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}

void ActionStandardSystem::doInsideMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.index = srcState.index + 1;
    dstState.prevStatePtr = &srcState;
    dstState.chunkedLabelIds.push_back(actionIdx2LabelIdx(transition.action));
    if (transition.action == nBegin || transition.action == nOutside) {
        if (dstState.onGoChunkIdx == -1) {
            dstState.onGoChunkIdx = 0;
        } else {
            dstState.prevChunkIdx = dstState.currChunkIdx;
            dstState.currChunkIdx = dstState.onGoChunkIdx;
            dstState.onGoChunkIdx = dstState.index;
        }
        dstState.prevChunkIdx = dstState.currChunkIdx;
        dstState.currChunkIdx = dstState.index;
    }
    dstState.lastAction = const_cast<CScoredTransition &>(transition).action;
    dstState.sentLength = srcState.sentLength;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}

void ActionStandardSystem::doBeginMove(State &srcState, State &dstState, const CScoredTransition &transition) {
    dstState.index = srcState.index + 1;
    dstState.prevStatePtr = &srcState;
    dstState.chunkedLabelIds.push_back(actionIdx2LabelIdx(transition.action));
    if (transition.action == nBegin || transition.action == nOutside) {
        if (dstState.onGoChunkIdx == -1) {
            dstState.onGoChunkIdx = 0;
        } else {
            dstState.prevChunkIdx = dstState.currChunkIdx;
            dstState.currChunkIdx = dstState.onGoChunkIdx;
            dstState.onGoChunkIdx = dstState.index;
        }
        dstState.prevChunkIdx = dstState.currChunkIdx;
        dstState.currChunkIdx = dstState.index;
    }
    dstState.lastAction = const_cast<CScoredTransition &>(transition).action;
    dstState.sentLength = srcState.sentLength;
    dstState.score = const_cast<CScoredTransition &>(transition).score;
}

void ActionStandardSystem::saveActionSystem(std::ostream &os) {
    labelManager.saveLabelManager(os);
    os << "labelSize" << " " << knowLabels.size() << std::endl;
    os << "nActNum" << " " << nActNum << std::endl;
    os << "nOutside" << " " << nOutside << std::endl;
    os << "nBegin" << " " << nBegin << std::endl;
}

void ActionStandardSystem::loadActionSystem(std::istream &is) {
    labelManager.loadLabelManager(is);
    std::string line;
    getline(is, line);
    std::istringstream iss(line);
    std::string tmp;
    int size;
    iss >> tmp >> size;
    for (int i = 0; i < size; i++) {
        getline(is, line);

        std::istringstream labelIss(line);
        std::string label;
        labelIss >> label;
        knowLabels.push_back(label);
    }

    getline(is, line);
    iss.str(line);
    iss >> tmp >> size;
    nActNum = size;
    
    getline(is, line);
    iss.str(line);
    iss >> tmp >> size;
    nOutside = size;

    getline(is, line);
    iss.str(line);
    iss >> tmp >> size;
    nBegin = size;
}
