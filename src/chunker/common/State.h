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
    int m_nIndex;
    State *previous_;
    std::vector<int> frontLabels;
    int last_action;
    int m_nLen;
    int beamIdx;
    double score;
    bool bGold;

    State(){
        bGold = true;
        beamIdx = 0;
        clear();
    }

    ~State() {}

    State(const State &s) : frontLabels(s.frontLabels){
        m_nIndex = s.m_nIndex;
        previous_ = s.previous_;
        last_action = s.last_action;
        m_nLen = s.m_nLen;
        score = s.score;
        beamIdx = s.beamIdx;
        bGold = s.bGold;
    }

    State& operator= (const State &s) {
        if (this == &s) {
            return *this;
        }

        this->m_nIndex = s.m_nIndex;
        this->previous_ = s.previous_;
        frontLabels = s.frontLabels;
        this->last_action = s.last_action;
        this->m_nLen = s.m_nLen;
        this->score = s.score;
        this->beamIdx = s.beamIdx;
        this->bGold = s.bGold;

        return *this;
    }

    bool complete() {
        return m_nIndex == m_nLen - 1;
    }

    void setBeamIdx(int idx) {
        beamIdx = idx;
    }

    void clear() {
        m_nIndex = -1;
        previous_ = nullptr;
        score = 0;
        last_action = -1;
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
