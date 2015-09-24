/*************************************************************************
	> File Name: src/depparser/BeamDecodor.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 18/09/15 17:03:00
 ************************************************************************/

#ifdef DEPPARSER_BEAMDECODER_H
#define DEPPARSER_BEAMDECODER_H


#include "Beam.h"
#include "State.h"
#include "TNNets.h"
#include "Instance.h"

class BeamDecodor{

public:
    bool bTrain;
    bool bEarlyUpdated;
    Beam beam;
    State * lattice;
    State ** lattice_index;
    CScoredTransition goldScoredTran;
    int nGoldTransitIndex;
    int nMaxLatticeSize;
    int nRound;
    int nMaxRound;
    Instance * inst;
   
    BeamDecodor(Instance * inst, int beamSize, bool bTrain) : beam(beamSize) {
        nMaxRound = 2 * ( input->size() -1 ); // remove the root in the input
        nMaxLatticeSize =  (beamSize + 1) * nMaxRound;
        nCorrectStateIndex = 0;
        nRound = 0;
        this->inst = inst;

        this->bTrain = bTrain;
        bEarlyUpdated = false;

        // construct the lattice
        lattice = new State[mnMaxLatticeSize];
        lattice_index = new State* [nMaxRound];
        correctState = lattice;
    }

    ~BeamDecodor(){
        delete[] lattice;
        delete[] lattice_index;
    }

    /*
     * beam search decoding
     */
    State * decoding( TNNet &tnnet, FeatureEmbedding<cpu> &fEmb,
                   FeatureExtractor & featExtract, GlobalExample * example = nullptr){

        State * retval;
        float maxScore = 0;
        bool bBeamContainGold;
        /*
         * initialize every lattice for decoding
         */
        for (int i = 0; i <= mnMaxLatticeSize; ++i)
            lattice[i].len_ = sentLen;
        lattice[0].clear();

        if(bTrain){
            lattice[0].setBeamIdx(0); // to know which neural net this state is generated from,
                                      // used for batch updating in the end of traning
            correctState = lattice; // the initial state is gold state
        }
        
        /*
         * set lattice_index as point to states in lattice
         * it's point to point
         */
        lattice_index[0] = lattice;
        lattice_index[1] = lattice_index[0] + 1;

        //set up mshadow tensor
        InitTensorEngine<gpu>();

        // for every round in training
        for(nRound = 1; nRound < nMaxRound; nRound++){

            // temp input layer
            TensorContainer<cpu, 2> input;
            input.Resize( Shape2( beam.beamFullSize, num_in ) );
            // temp output layer
            TensorContainer<cpu, 2> pred;
            pred.Resize( Shape2( beam.beamFullSize, num_out ) );

            tnnet.genNextStepNet();

            std::vector<std::vector<int> > featureVectors( nRound == 1 ? 1 : beam.currentBeamSize); // extract feature vectors in batch
            getInputBatch(lattice_index[ nRound - 1 ], inst->wordIdx,
                          inst->tagIdx, featureVectors);
            fEmb.returnInput(featureVectors, input);
            net->Forward(input, pred);

            // for every state in the last beam, expand and insert into next beam
            int stateIdx = 0;
            for (State * currentState = lattice_index[nRound - 1];
                    currentState != lattice_index[nRound]; ++currentState, ++stateIdx) {
                std::vector<int> validActs;
                currentState->getValidActs(validActs);

                //for every valid action
                for(unsigned actID = 0; actID < validActs.size(); ++actID){
                    //skip invalid action
                    if(validActs[actID] == -1)
                        continue;
                    //construct scored transition, and insert into beam
                    CScoredTransition trans(currentState, actID, currentState->score + pred[stateIdx][actID]);
                    int insertd = beam.insert(trans);

                    if( currentState->bGold == true && actID == example->goldActs[nRound - 1] ){ // if this is gold transition
                        goldScoredTran = trans;
                        bEarlyUpdate = inserted == 0; // if do not insert the gold into the beam, then early update
                        nGoldTransitIndex = actID;
                    }
                } // valid action #for end

            } // expand current step states and inset them into beam #for end

            if ( bEarlyUpdate && bTrain)
               break;

            float dMaxScore = 0;
            //lazy expand the states in the beam
            for (int i = 0; i < beam.currentBeamSize; ++i) {
                const CScoredTransition& transition = beam.beam[i];
                State* target = lattice_index[nRound] + i;
                target->copy( *(transition.source) );
                // generate candidate state according to the states in beam
                target->Move(transition.action);
                target->setBeamIdx(i);
                target->score = transition.score;
                target->previous_ = transition.source;
                
                if( i == 0 || target->score > dMaxScore ){
                    dMaxScore = target->score;
                    retval = target;
                }
            }

            // prepare lattice for next parsing round
            lattice_index[nRound + 1] = lattice_index[nRound] + beam.currentBeamSize;
        } //round #for end

        ShutdownTensorEngine<gpu>();

        return retval; // return without early updating
    }

    /*
     * get the feature vector in all the beam states,
     * and return the input layer of neural network in a batch.
     */
    void getInputBatch(State* state, std::vector<int>& wordIndexCache,
                       std::vector<int>& tagIndexCache,
                       std::vector<std::vector<int> >& featvecs) {
  
        for(unsigned i = 0; i < featvecs.size(); i++){
            std::vector<int> featvec(CConfig::nFeatureNum);
            featExtractor.featureExtract( state + i, wordIndexCache, tagIndexCache, featvec);
            featvecs.push_back(featvec);
        }
    }

    int getGoldAct( GlobalExample * ge ){
        return ge->goldActs[ nRound - 1 ];
    }
    
};

#endif
