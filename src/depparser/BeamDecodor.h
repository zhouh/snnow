/*************************************************************************
  > File Name: src/depparser/BeamDecodor.h
  > Author: Hao Zhou
  > Mail: haozhou0806@gmail.com 
  > Created Time: 18/09/15 17:03:00
 ************************************************************************/

#ifndef DEPPARSER_BEAMDECODER_H
#define DEPPARSER_BEAMDECODER_H

#include "Beam.h"
#include "State.h"
#include "TNNets.h"
#include "Instance.h"
#include "ArcStandardSystem.h"

#include "mshadow/tensor.h"
#include "FeatureEmbedding.h"

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
        int nSentLen;
        Instance * inst;

        BeamDecodor(Instance * inst, int beamSize, bool bTrain) : beam(beamSize) {
            nSentLen = inst->input.size();
            nMaxRound = 2 * ( nSentLen - 1 ); // remove the root in the input
            nMaxLatticeSize = (beamSize + 1) * nMaxRound;
            nRound = 0;
            this->inst = inst;

            this->bTrain = bTrain;
            bEarlyUpdated = false;

            // construct the lattice
            lattice = new State[nMaxLatticeSize];
            lattice_index = new State* [nMaxRound + 2];
        }

        ~BeamDecodor(){
            //std::cout<<"begin to free"<<std::endl;
            delete[] lattice;
            delete[] lattice_index;
        }

        /*
         * beam search decoding
         */
        State * decoding( ArcStandardSystem * tranSystem, TNNets &tnnet, FeatureExtractor & featExtract, 
                FeatureEmbedding & fEmb, GlobalExample * example = nullptr){

            State * retval = nullptr;
            /*
             * initialize every lattice for decoding
             */
            for (int i = 0; i < nMaxLatticeSize; ++i){

                lattice[i].len_ = nSentLen;
                lattice[i].initCache();
            }

            lattice[0].clear();

            if(bTrain)
                lattice[0].setBeamIdx(0); // to know which neural net this state is generated from,
            // used for batch updating in the end of traning

            /*
             * set lattice_index as point to states in lattice
             * it's point to point
             */
            lattice_index[0] = lattice;
            lattice_index[1] = lattice_index[0] + 1;

            //set up mshadow tensor
            InitTensorEngine<XPU>();

            // for every round in training
            for(nRound = 1; nRound <= nMaxRound; nRound++){

                //std::cout<<"round\t"<<nRound << "of MaxRound"<< nMaxRound<<std::endl;

                // temp input layer
                TensorContainer<cpu, 2, real_t> input;
                input.Resize( Shape2( beam.beamFullSize, tnnet.num_in ) );
                // temp output layer
                TensorContainer<cpu, 2, real_t> pred;
                pred.Resize( Shape2( beam.beamFullSize, tnnet.num_out ) );

                //std::cout<<"build next neural net"<<std::endl;
                /*
                 * In the training process, we need a new neural net to forward, 
                 * with which, we can directly update parameters in the end other than forward and update!
                 */
                if( bTrain )
                    tnnet.genNextStepNet();

                /*
                 * extract features and generate input embedding
                 */
                std::vector<std::vector<int> > featureVectors; // extract feature vectors in batch
                featureVectors.resize(nRound == 1 ? 1 : beam.currentBeamSize);
                getInputBatch(featExtract, lattice_index[ nRound - 1 ], inst->wordCache,
                        inst->tagCache, featureVectors);
                fEmb.returnInput(featureVectors, input);

                //std::cout<<"begin forward"<<std::endl;
                tnnet.Forward(input, pred);
                //for(int i = 0; i < pred.size(0); i++){ 
                    //for(int j = 0; j < pred.size(1); j++)
                        //std::cout<<pred[i][j]<<" ";
                    //std::cout<<std::endl;
                //}

                /*
                 * clear the beam and prepare for the next beam expand
                 */
                beam.clear();

                //std::cout<<"expand the states in the beam"<<std::endl;
                // for every state in the last beam, expand and insert into next beam
                int stateIdx = 0;
                for (State * currentState = lattice_index[nRound - 1];
                        currentState != lattice_index[nRound]; ++currentState, ++stateIdx) {
                    std::vector<int> validActs;
                    tranSystem->getValidActs(*currentState, validActs);

                    bool noneValid = true;
                    //for every valid action
                    for(unsigned actID = 0; actID < validActs.size(); ++actID){
                        //skip invalid action
                        if(validActs[actID] == -1)
                            continue;
                        noneValid = false;
                        //construct scored transition, and insert into beam
                        CScoredTransition trans;
                        trans(currentState, actID, currentState->score + pred[stateIdx][actID]);
                        int inserted = beam.insert(trans);

                        if( bTrain && currentState->bGold == true 
                                && actID == example->goldActs[nRound - 1] ){ // if this is gold transition
                            goldScoredTran = trans;
                            bEarlyUpdated = inserted == 0; // if do not insert the gold into the beam, then early update
                            nGoldTransitIndex = actID;
                        }
                    } // valid action #for end
                    assert(noneValid == false);

                } // expand current step states and inset them into beam #for end

                if ( bEarlyUpdated && bTrain){
                    std::cout<<"early update! round "<< nRound << " of maxRound "<< nMaxRound <<std::endl;
                    break;
                }

                //std::cout<<"lazy update"<<std::endl;
                float dMaxScore = 0;
                //lazy expand the states in the beam
                for (int i = 0; i < beam.currentBeamSize; ++i) {
                    const CScoredTransition& transition = beam.beam[i];
                    State* target = lattice_index[nRound] + i;
                    target->copy( *(transition.source) );
                    // generate candidate state according to the states in beam
                    tranSystem->Move(*target, transition.action);
                    if(bTrain){
                        target->setBeamIdx(i);
                        target->bGold = transition.source->bGold 
                            && transition.action == example->goldActs[ nRound - 1 ];
                    } 
                    target->score = transition.score;
                    target->previous_ = transition.source;

                    if( i == 0 || target->score > dMaxScore ){
                        dMaxScore = target->score;
                        retval = target;
                    }
                }

                // prepare lattice for next parsing round
                lattice_index[nRound + 1] = lattice_index[nRound] + beam.currentBeamSize;
                //std::cout<<"end of round";
            } //round #for end

            ShutdownTensorEngine<XPU>();

            return retval; // return without early updating
        }

        /*
         * get the feature vector in all the beam states,
         * and return the input layer of neural network in a batch.
         */
        void getInputBatch(FeatureExtractor & featExtractor, State* state, std::vector<int>& wordIndexCache,
                std::vector<int>& tagIndexCache,
                std::vector<std::vector<int> >& featvecs) {

            for(unsigned i = 0; i < featvecs.size(); i++){
                //std::cout<<" feature vector:\t "<<featvecs.size()<<"\t"<<i<<std::endl;
                featvecs[i].resize(CConfig::nFeatureNum);
                featExtractor.featureExtract( state + i, wordIndexCache, tagIndexCache, featvecs[i]);
            }
        }

        int getGoldAct( GlobalExample * ge ){
            return ge->goldActs[ nRound - 1 ];
        }
};

#endif
