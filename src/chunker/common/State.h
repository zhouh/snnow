/*************************************************************************
	> File Name: State.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 08:33:52 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_STATE_H_
#define _CHUNKER_COMMON_STATE_H_

#include <vector>

class State {
public:
    int index;
    State *prevStatePtr;
    std::vector<int> chunkedLabelIds;
    int lastAction;
    int sentLength;
    int beamIdx;
    double score;
    bool bGold;
    int onGoChunkIdx;
    int currChunkIdx;
    int prevChunkIdx;

    State(){
        bGold = true;
        beamIdx = 0;
        clear();
    }

    ~State() {}

    State(const State &s) : chunkedLabelIds(s.chunkedLabelIds){
        index = s.index;
        prevStatePtr = s.prevStatePtr;
        lastAction = s.lastAction;
        sentLength = s.sentLength;
        score = s.score;
        beamIdx = s.beamIdx;
        bGold = s.bGold;
        onGoChunkIdx = s.onGoChunkIdx;
        currChunkIdx = s.currChunkIdx;
        prevChunkIdx = s.prevChunkIdx;
    }

    State& operator= (const State &s) {
        if (this == &s) {
            return *this;
        }

        this->index = s.index;
        this->prevStatePtr = s.prevStatePtr;
        this->chunkedLabelIds = s.chunkedLabelIds;
        this->lastAction = s.lastAction;
        this->sentLength = s.sentLength;
        this->score = s.score;
        this->beamIdx = s.beamIdx;
        this->bGold = s.bGold;
        this->onGoChunkIdx = s.onGoChunkIdx;
        this->currChunkIdx = s.currChunkIdx;
        this->prevChunkIdx = s.prevChunkIdx;

        return *this;
    }

    bool complete() {
        return index == sentLength - 1;
    }

    void setBeamIdx(int idx) {
        beamIdx = idx;
    }

    void clear() {
        index = -1;
        prevStatePtr = nullptr;
        score = 0;
        lastAction = -1;

        onGoChunkIdx = -1;
        currChunkIdx = -1;
        prevChunkIdx = -1;
    }
};

struct CScoredTransition {
    //! The pointer to the source state
    State *source;

    //! The action applied to the source state
    int action;

    //! The result score
    double score;

    //! If this transition is gold
    bool bGold;

    CScoredTransition(): source(nullptr), action(-1), score(0), bGold(false) {}

    CScoredTransition(State *s, int a, double sc): source(s), action(a), score(sc), bGold(false) {}

    CScoredTransition(const CScoredTransition &s) {
        this->source = s.source;
        this->action = s.action;
        this->score = s.score;
        this->bGold = s.bGold;
    }

    CScoredTransition& operator= (const CScoredTransition &s) {
        if (this == &s) {
            return *this;
        }

        this->source = s.source;
        this->action = s.action;
        this->score = s.score;
        this->bGold = s.bGold;

        return *this;
    }

    ~CScoredTransition() {}

    void operator()(State *s, int a, double sc) {
        source = s;
        action = a;
        score = sc;

        bGold = false;
    }

};

#endif
