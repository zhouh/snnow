/*************************************************************************
	> File Name: Beam.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 24 Nov 2015 01:11:05 PM CST
 ************************************************************************/
#include "Beam.h"

int Beam::insert(const CScoredTransition &transition) {
    if (currentBeamSize == beamFullSize) {
        if (transition.score > beam[0].score) {
            std::pop_heap(beam, beam + currentBeamSize, ScoredTransitionMore);
            beam[currentBeamSize - 1] = transition;
            std::push_heap(beam, beam + currentBeamSize, ScoredTransitionMore);

            return 1;
        } else {
            return 0;
        }
    }

    beam[currentBeamSize] = transition;
    ++currentBeamSize;
    std::push_heap(beam, beam + currentBeamSize, ScoredTransitionMore);

    return 1;
}

float Beam::getMaxScoreInBeam() {
    float maxScore = 0;

    if (currentBeamSize == 0) {
        return maxScore;
    }

    maxScore = beam[0].score;

    for (int i = 1; i < currentBeamSize; i++) {
        if (beam[i].score > maxScore) {
            maxScore = beam[i].score;
        }
    }

    return maxScore;
}

bool ScoredTransitionMore(const CScoredTransition &x, const CScoredTransition &y) {
    return x.score > y.score;
}
