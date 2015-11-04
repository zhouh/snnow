/*************************************************************************
	> File Name: src/include/Beam.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 08/09/15 10:52:59
 ************************************************************************/

#ifndef BEAM_H_
#define BEAM_H_

#include<algorithm>

#include "State.h"
#include "DepTree.h"

/** 
 * std::heap is a max heap, and the compare function return first element is less that the second.
 * But here, we need a min heap, so return x.score > y.score
 */
bool
ScoredTransitionMore(const CScoredTransition& x, const CScoredTransition& y);

class Beam{

    public:
    CScoredTransition* beam;
    int beamFullSize;
    int currentBeamSize;
    bool bBeamContainGoldState;

    Beam(int beamFullSize){
        this->beamFullSize = beamFullSize;
        this->currentBeamSize = 0;
        beam = new CScoredTransition[beamFullSize];
        bBeamContainGoldState = false;
    }

    void clear(){

        currentBeamSize = 0;
        bBeamContainGoldState = false;
    }

    ~Beam(){
        delete[] beam; 
    }

    /* Insert one transition into beam,
     *  if insert successfully, return 1, else return o
     */
   inline int insert(const CScoredTransition& transition){
        //beam is full
        if (currentBeamSize == beamFullSize) {
            if (transition.score > beam[0].score) {
              std::pop_heap(beam, beam + currentBeamSize,
                  ScoredTransitionMore);
              beam[currentBeamSize - 1] = transition;
              std::push_heap(beam, beam + currentBeamSize,
                  ScoredTransitionMore);
              return 1;
            }
            else
                return 0;
        }

        //beam not full, insert directly
        beam[currentBeamSize] = transition;
        std::push_heap(beam, beam + currentBeamSize + 1,
                       ScoredTransitionMore);
        ++ currentBeamSize;
        return 1;
    }
    
    /*
     * return the max score in the beam
     */
    inline float getMaxScoreInBeam(){
        float maxScore = 0;
        for( int i = 0; i < currentBeamSize; i++ )
            if( i == 0 || beam[i].score > maxScore )
               maxScore = beam[i].score;
        return maxScore;
    }
    
};


#endif
