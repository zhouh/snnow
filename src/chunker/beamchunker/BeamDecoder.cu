/*************************************************************************
	> File Name: BeamDecoder.cu
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 03:59:17 PM CST
 ************************************************************************/
#include "TNNets.h"
#include "mshadow/tensor.h"

#include "BeamDecoder.h"

BeamDecoder::BeamDecoder(Instance *inst, 
            std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
            std::shared_ptr<FeatureManager> featureMangerPtr,
            std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
            int beamSize, 
            bool bTrain) : 
            m_transSystemPtr(transitionSystemPtr),
            m_featManagerPtr(featureMangerPtr),
            m_featEmbManagerPtr(featureEmbManagerPtr),
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

BeamDecoder::BeamDecoder(Instance *inst, 
            std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
            std::shared_ptr<FeatureManager> featureMangerPtr,
            std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
            int beamSize, 
            State *lattice,
            State **lattice_index,
            bool bTrain) : 
            m_transSystemPtr(transitionSystemPtr),
            m_featManagerPtr(featureMangerPtr),
            m_featEmbManagerPtr(featureEmbManagerPtr),
            beam(beamSize) {
    nSentLen = inst->input.size();
    nMaxRound = nSentLen;

    nMaxLatticeSize = (beamSize + 1) * nMaxRound;
    nRound = 0;

    this->inst = inst;
    this->bTrain = bTrain;

    bEarlyUpdate = false;
   
    this->lattice = lattice;
    this->lattice_index = lattice_index; 
    // lattice = new State[nMaxLatticeSize];
    // lattice_index = new State *[nMaxRound + 2];
}

BeamDecoder::~BeamDecoder() {
    if (!bTrain) {
        delete []lattice;
        delete []lattice_index;
    }
}

void BeamDecoder::generateLabeledSequence(TNNets &tnnets, LabeledSequence &predictedSent) {

    State *predState = decode(tnnets);

    m_transSystemPtr->generateOutput(*predState, predictedSent);
}

State* BeamDecoder::decode(TNNets &tnnet, GlobalExample *gExample) {
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

    // input layer and predicted layer for each round
    TensorContainer<cpu, 2, real_t> input;
    input.Resize(Shape2(beam.beamFullSize, tnnet.num_in));

    TensorContainer<cpu, 2, real_t> pred;
    pred.Resize(Shape2(beam.beamFullSize, tnnet.num_out));

    // for every round in training
    for (nRound = 1; nRound <= nMaxRound; nRound++) {
        input = 0.0;
        pred = 0.0;

        // In the training process, we need a new neural net to forward, 
        // with which, we can directly update parameters in the end other
        // than forwarding and updating!
        if (bTrain) {
            tnnet.genNextStepNet();
        }

        // extract features and generate input embeddings
        // std::vector<std::vector<int>> featureVectors; // extracted feature vectors in batch.
        std::vector<FeatureVector> featureVectors;
        featureVectors.resize(nRound == 1 ? 1 : beam.currentBeamSize);
        generateInputBatch(lattice_index[nRound - 1], inst, featureVectors);
        if (bTrain) {
            tnnet.addFeatVecs(featureVectors);
        }
        m_featEmbManagerPtr->returnInput(featureVectors, tnnet.modelParas->featEmbs, input);

        tnnet.Forward(input, pred);

        // clear the beam for the next beam expand
        beam.clear();

        // for each state in the latest beam, expand it and insert expanded state into next beam
        int stateIdx = 0;
        for (State *currentState = lattice_index[nRound - 1]; currentState != lattice_index[nRound]; ++currentState, ++stateIdx) {
            std::vector<int> validActs;
            m_transSystemPtr->generateValidActs(*currentState, validActs);

            bool noValid = true;
            // for each valid action
            for (unsigned actId = 0; actId < validActs.size(); ++actId) {
                // skip invalid action
                if (validActs[actId] == -1) {
                    continue;
                }

                noValid = false;
                // construct scored transition and insert it into beam
                CScoredTransition trans(currentState, actId, currentState->score + pred[stateIdx][actId]); // TODO: ignore inValid scores ?
                
                // if this is the gold transition
                if (bTrain && currentState->bGold && actId == gExample->goldActs[nRound - 1]) {
                    trans.bGold = true;
                    goldScoredTran = trans;
                }
                beam.insert(trans);
            }
            assert (noValid == false);
        }

        bEarlyUpdate = true; // early update if gold transition was not inserted into the beam
        for (int i = 0; i < beam.currentBeamSize; ++i) {
            if (beam.beam[i].bGold) {
                bEarlyUpdate = false;
            }
        }

        if (bTrain && bEarlyUpdate) {
            break;
        }

        float dMaxScore = 0.0;
        // lazy expand the target states in the beam
        for (int i = 0; i < beam.currentBeamSize; ++i) {
            const CScoredTransition transition = beam.beam[i];

            State *target = lattice_index[nRound] + i;
            *target = *(transition.source);
            m_transSystemPtr->move(*(transition.source), *target, transition);

            if (bTrain) {
                target->bGold = transition.bGold;
                target->setBeamIdx(i);       // the corresponding nnet to be forwarded in the tnnets of specific round
                if (transition.bGold) {
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

    return retval; // return without early updating
}
