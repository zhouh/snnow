/*************************************************************************
	> File Name: BeamDecoder.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 03 Dec 2015 10:26:06 AM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMDECODER_H_
#define _CHUNKER_BEAMDECODER_H_

#include "Beam.h"
#include "State.h"
#include "Instance.h"
#include "ActionStandardSystem.h"

#include "mshadow/tensor.h"
#include "FeatureEmbedding.h"

class BeamDecoder {
public:
    bool bTrain;
    bool bEarlyUpdate;
    Beam beam;
    State * lattice;
    State ** lattice_index;
    CScoredTransition goldScoredTran;
    
    int nGoldTransitionIndex;
    int nMaxLatticeSize;
    int nRound;
    int nMaxRound;
    int nSentLen;

    Instance * inst;

    BeamDecoder(Instance *inst, int beamSize, bool bTrain) : beam(beamSize) {
        nSentLen = inst.input.size();
        nMaxRound = nSentLen;

        nMaxLatticeSize = (beamSize + 1) * nMaxRound;
        nRound = 0;

        this->inst = inst;
        this->bTrain = bTrain;

        bEarlyUpdate = false;
        
        lattice = new State[nMaxLatticeSize];
        lattice_index = new State *[nMaxRound + 2];
    }

    ~BeamDecoder() {
        delete []lattice;
        delete []lattice_index;
    }

    State *decode(ActionStandardSystem &tranSystem, TNNets &tnnet, FeatureExtractor &featExtractor, FeatureEmbedding &fEmb, GlobalExample *gExample = nullptr) {
        State *retval = nullptr;

        for (int i = 0; i < nMaxLatticeSize; ++i) {
            lattice[i].m_nLen = nSentLen;
        }

        if (bTrain) {
            // to know which neural net this state is generated from,
            // which is used for batch updating in the end of training
            lattice[0].setBeamIdx(0); 
        }
    }
};

#endif
