/*************************************************************************
	> File Name: Beam.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Mon 23 Nov 2015 01:31:58 PM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMCHUNKER_BEAM_H_
#define _CHUNKER_BEAMCHUNKER_BEAM_H_

#include <algorithm>

#include "State.h"

bool ScoredTransitionMore(const CScoredTransition &x, const CScoredTransition &y);

class Beam {
public:
    CScoredTransition *beam;  
    int beamFullSize;
    int currentBeamSize;
    bool bBeamContainGoldState;

    Beam(int beamFullSize) {
        this->beamFullSize = beamFullSize;
        this->currentBeamSize = 0;
        beam = new CScoredTransition[beamFullSize];
        bBeamContainGoldState = false;
    }

    ~Beam() {
        delete []beam;
    }

    void clear() {
        currentBeamSize = 0;
        bBeamContainGoldState = false;
    }

    int insert(const CScoredTransition &transition);

    float getMaxScoreInBeam();
};

#endif
