//
// Created by zhouh on 16-4-1.
//

#include "Beam.h"

int Beam::insert(const CScoredTransition& transition) {
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

bool Beam::isMaxScoreGold() {
    float maxScore = 0;
    int maxIndex;

    if (currentBeamSize == 0) {
        return false;
    }

    maxScore = beam[0].score;
    maxIndex = 0;

    for (int i = 1; i < currentBeamSize; i++) {
        if (beam[i].score > maxScore) {
            maxScore = beam[i].score;
            maxIndex = i;
        }
    }

    return beam[maxIndex].bGold;
}

/**
 *  compare the ScoredTransition
 *  If the score of x is larger than y, return true
 */
bool ScoredTransitionCompare(const CScoredTransition &x, const CScoredTransition &y) {
    return x.score > y.score;
}