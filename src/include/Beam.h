//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_BEAM_H
#define SNNOW_BEAM_H


#include <algorithm>
#include "/src/include/base/State.h"

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

    bool isMaxScoreGold();
};


#endif //SNNOW_BEAM_H
