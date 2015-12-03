/*************************************************************************
	> File Name: State.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 08:33:52 PM CST
 ************************************************************************/
#ifndef _CHUNKER_STATE_H_
#define _CHUNKER_STATE_H_

class State {
public:
    int m_nIndex;
    State *previous_;
    int last_action;
    int m_nLen;
    int beamIdx;
    double score;

    State(){
        clear();
    }

    ~State() {}

    State(const State &s) {
        m_nIndex = s.m_nIndex;
        previous_ = s.previous_;
        last_action = s.last_action;
        m_nLen = s.m_nLen;
        score = s.score;
        beamIdx = s.beamIdx;
    }

    State& operator= (State &s) {
        if (this == &s) {
            return *this;
        }

        this->m_nIndex = s.m_nIndex;
        this->previous_ = s.previous_;
        this->last_action = s.last_action;
        this->m_nLen = s.m_nLen;
        this->score = s.score;
        this->beamIdx = s.beamIdx;

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
        previous_ = NULL;
        score = 0;
        last_action = -1;
        m_nLen = 0;
        beamIdx = 0;
    }
};

struct CScoredTransition {
    //! The pointer to the source state
    const State *source;

    //! The action applied to the source state
    int action;

    //! The result score
    double score;

    CScoredTransition(): source(NULL), action(-1), score(0) {}

    CScoredTransition(const State *s, int a, double sc): source(s), action(a), score(sc) {}

    void operator()(const State *s, int a, double sc) {
        source = s;
        action = a;
        score = sc;
    }
};

#endif
