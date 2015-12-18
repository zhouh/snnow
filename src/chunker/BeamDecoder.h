/*************************************************************************
	> File Name: BeamDecoder.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 03 Dec 2015 10:26:06 AM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMDECODER_H_
#define _CHUNKER_BEAMDECODER_H_

#include <limits.h>
#include <assert.h>

#include "Beam.h"
#include "State.h"
#include "TNNets.h"
#include "Instance.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"

#include "mshadow/tensor.h"
#include "FeatureEmbedding.h"
#include "FeatureExtractor.h"

class BeamDecoder {
public:
    ActionStandardSystem *tranSystem;
    FeatureExtractor *featExtractor;
    FeatureEmbedding *fEmb;

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

    BeamDecoder(Instance *inst, 
                ActionStandardSystem *transitionSystem, 
                FeatureExtractor *featureExtractor, 
                FeatureEmbedding *featureEmbedding, 
                int beamSize, 
                bool bTrain) : 
                tranSystem(transitionSystem), 
                featExtractor(featureExtractor), 
                fEmb(featureEmbedding), 
                beam(beamSize) {
        nSentLen = inst->input.size();
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

    void generateChunkedSentence(TNNets &tnnets, ChunkedSentence &predictedSent) {
        // std::cout << "[input]: " << std::endl;
        // std::cout << predictedSent << std::endl;

        State *predState = decode(tnnets);

        // std::cout << std::endl;
        // std::cout << "current beam size: " << beam.currentBeamSize << std::endl;
        // std::cout << predictedSent;

        // std::vector<int> predictedActions;
        // State *ptr = predState;
        // if (ptr == nullptr) {
        //     std::cout << "predstate is nullptr!" << std::endl;
        // }
        // int i = 1;
        // while (ptr != nullptr && ptr->last_action != -1) {
        //     i++;
        //     predictedActions.push_back(ptr->last_action);
        //     ptr = ptr->previous_;
        // }
        // std::cout << "decoded path length: " << i << std::endl;
        // std::cout << "[pred action sequences]: ";
        // for (int i = 0; i < predictedActions.size(); i++) {
        //     std::cout << predictedActions[predictedActions.size() - 1 - i] << " ";
        // }
        // std::cout << std::endl;

        tranSystem->generateOutput(*predState, predictedSent);
    }

    State *decode(TNNets &tnnet, GlobalExample *gExample = nullptr) {
        State *retval = nullptr;

        for (int i = 0; i < nMaxLatticeSize; ++i) {
            lattice[i].m_nLen = nSentLen;
        }

        if (bTrain) {
            // to know which neural net this state is generated from,
            // which is used for batch updating in the end of training
            lattice[0].setBeamIdx(0); 
        }

        lattice[0].clear();

        //  set  lattice_index as point to states in lattice
        lattice_index[0] = lattice;
        lattice_index[1] = lattice_index[0] + 1;

        // set up mshadow tensor
        InitTensorEngine<XPU>();
        // for every round in training
        for (nRound = 1; nRound <= nMaxRound; nRound++) {
            // temporary input layer
            TensorContainer<cpu, 2, real_t> input;
            input.Resize(Shape2(beam.beamFullSize, tnnet.num_in));
            
            // temporary output layer
            TensorContainer<cpu, 2, real_t> pred;
            pred.Resize(Shape2(beam.beamFullSize, tnnet.num_out));

            // In the training process, we need a new neural net to forward, 
            // with which, we can directly update parameters in the end other
            // than forwarding and updating!
            if (bTrain) {
                tnnet.genNextStepNet();
            }

            // extract features and generate input embeddings
            std::vector<std::vector<int>> featureVectors; // extracted feature vectors in batch
            featureVectors.resize(nRound == 1 ? 1 : beam.currentBeamSize);
            generateInputBatch(*featExtractor, lattice_index[nRound - 1], *(inst), featureVectors);
            fEmb->returnInput(featureVectors, input);

            tnnet.Forward(input, pred);

            // clear the beam for the next beam expand
            beam.clear();

            // for each state in the latest beam, expand it and insert expanded state into next beam
            int stateIdx = 0;
            for (State *currentState = lattice_index[nRound - 1]; currentState != lattice_index[nRound]; ++currentState, ++stateIdx) {
                std::vector<int> validActs;
                tranSystem->generateValidActs(*currentState, validActs);

                bool noValid = true;
                // for each valid action
                for (unsigned actId = 0; actId < validActs.size(); ++actId) {
                    // skip invalid action
                    if (validActs[actId] == -1) {
                        continue;
                    }

                    noValid = false;
                    // construct scored transition and insert it into beam
                    CScoredTransition trans;
                    trans(currentState, actId, currentState->score + pred[stateIdx][actId]); // TODO: ignore inValid scores ?
                    
                    if (isnan((real_t)pred[stateIdx][actId])) {
                        std::cout << "found a nan" << std::endl;
                    }
                    int inserted = beam.insert(trans);
                    // if this is the gold transition
                    if (bTrain && currentState->bGold && actId == gExample->goldActs[nRound - 1]) {
                        goldScoredTran = trans;
                        bEarlyUpdate = (inserted == 0); // early update if gold transition was not inserted into the beam
                        // nGoldTransitionIndex = actId;   // TODO something is wrong ?
                    }
                }
                assert (noValid == false);
            }

             if (bEarlyUpdate && bTrain) {
                 std::cout << "early update at round " << nRound << " of maxRound " << nMaxRound << std::endl;
                 break;
             }

             float dMaxScore = 0.0;
             // lazy expand the target states in the beam
             for (int i = 0; i < beam.currentBeamSize; ++i) {
                 const CScoredTransition transition = beam.beam[i];

                 State *target = lattice_index[nRound] + i;
                 *target = *(transition.source);
                 tranSystem->move(*(transition.source), *target, transition);

                 if (bTrain) {
                     target->bGold = transition.source->bGold && transition.action == gExample->goldActs[nRound - 1];
                     target->setBeamIdx(i);       // the corresponding nnet to be forwarded in the tnnets of specific round
                     if (target->bGold) {
                         nGoldTransitionIndex = i;
                     }
                 }

                 if (i == 0 || target->score > dMaxScore) {
                     dMaxScore = target->score;
                     retval = target;
                 }
             }

             // prepare lattice for next chunking round
             lattice_index[nRound + 1] = lattice_index[nRound] + beam.currentBeamSize;
        }

        // shut down mshadow tensor engine
        ShutdownTensorEngine<XPU>();

        return retval; // return without early updating
    }

private:
    /* generate the feature vector in all the beam states,
     * and return the input layer of neural network in batch.
    */
    void generateInputBatch(FeatureExtractor &featExtractor, State *state, Instance &inst, std::vector<std::vector<int>> &featvecs) {
        for (int i = 0; i < featvecs.size(); i++) {
            featvecs[i].resize(CConfig::nFeatureNum);
            featExtractor.extractFeature(*(state + i), inst, featvecs[i]);
        }
    }
};

#endif