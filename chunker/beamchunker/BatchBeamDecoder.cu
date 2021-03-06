/*************************************************************************
	> File Name: BatchBeamDecoder.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 16 Jan 2016 03:10:51 PM CST
 ************************************************************************/
#include "TNNets.h"
#include "BatchBeamDecoder.h"

BatchBeamDecoderMemoryManager::BatchBeamDecoderMemoryManager(const int beamSize, const int decoderItemSize, const int longestLen, const int threadNum) :
    m_nThread(threadNum),
    m_nBeamSize(beamSize),
    m_nDecoderItemSize(decoderItemSize),
    lattices(threadNum, std::vector<State *>(decoderItemSize)),
    lattice_indexes(threadNum, std::vector<State **>(decoderItemSize))
{
    const int nMaxLatticeSize = (beamSize + 1) * longestLen;

    for (int i = 0; i < m_nThread; i++) {
        for (int j = 0; j < m_nDecoderItemSize; j++) {
            lattices[i][j] = new State[nMaxLatticeSize];
            lattice_indexes[i][j] = new State*[longestLen + 2];
        }
    }
}

BatchBeamDecoderMemoryManager::~BatchBeamDecoderMemoryManager() {
    for (int i = 0; i < m_nThread; i++) {
        for (int j = 0; j < m_nDecoderItemSize; j++) {
            delete []lattices[i][j];
            delete []lattice_indexes[i][j];
        }
    }
}

std::vector<State *> BatchBeamDecoder::decode(TNNets &tnnet, std::vector<GlobalExample *> &gExamplePtrs) {
    assert (m_bTrain);

    std::vector<State *> retvals(m_nInstSize, nullptr);

    for (int i = 0; i < m_nInstSize; i++) {
        State *s = m_lLatticePtrs[i];
        const int nMaxLatticeSize = m_lnMaxLatticeSizes[i];
        const int nSentLen = m_lnSentLens[i];

        for (int j = 0; j < nMaxLatticeSize; j++) {
            s[j].sentLength = nSentLen;
        }
    }

    for (int i = 0; i < m_nInstSize; i++) {
        m_lLatticePtrs[i][0].setBeamIdx(0);
        m_lLatticePtrs[i][0].clear();

        m_lLattice_indexPtrs[i][0] = m_lLatticePtrs[i];
        m_lLattice_indexPtrs[i][1] = m_lLattice_indexPtrs[i][0] + 1;
    }

    const int batch_size = m_nInstSize * m_nBeamSize;
    const int num_in = tnnet.num_in;
    const int num_out = tnnet.num_out;

    TensorContainer<cpu, 2, real_t> input(Shape2(batch_size, num_in));
    TensorContainer<cpu, 2, real_t> pred(Shape2(batch_size, num_out));

    std::vector<bool> itemCompeleteds(m_nInstSize, false);
    auto checkCompleted = [&itemCompeleteds]() -> bool {
        for (auto b : itemCompeleteds) {
            if (!b) {
                return false;
            }
        }

        return true;
    };
    int nRound = 1;
    while (!checkCompleted()) {
        std::vector<FeatureVector> batchFeatureVectors;
        tnnet.moveToNextNet();
        for (int insti = 0; insti < m_nInstSize; insti++) {
            if (!itemCompeleteds[insti]) {
                m_lnExpandRounds[insti] = nRound;
            }
        }
        generateBatchInput(num_in, nRound, tnnet, itemCompeleteds, input, batchFeatureVectors);

        tnnet.addFeatVecs(batchFeatureVectors);

        pred = 0.0;
        // batch forward compute 
        // FOR CHECK
        tnnet.Forward(input, pred);

        generateBeams(pred, nRound, gExamplePtrs, itemCompeleteds);

        lazyExpandBeams(nRound, itemCompeleteds, retvals);

        // prepare lattice for next chunking round
        for (int insti = 0; insti < m_nInstSize; insti++) {
            if (itemCompeleteds[insti]) {
                continue;
            }

            m_lLattice_indexPtrs[insti][nRound + 1] = m_lLattice_indexPtrs[insti][nRound] + m_lBeamPtrs[insti]->currentBeamSize;
        }

        // check each brach if it is completed
        for (int insti = 0; insti < m_nInstSize; insti++) {
            if (nRound >= m_lnMaxRounds[insti]) {
                itemCompeleteds[insti] = true;
            }
        }

        nRound++;
    }

    return retvals;
}


void BatchBeamDecoder::generateBatchInput(const int num_in, const int nRound, const TNNets &tnnet, const std::vector<bool> &itemCompeleteds, TensorContainer<cpu, 2, real_t> &input, std::vector<FeatureVector> &batchFeatureVectors) {
    input = 0.0;

    int input_index = 0;
    // fill full in the input
    for (int insti = 0; insti < m_nInstSize; insti++, input_index += m_nBeamSize) {
        if (itemCompeleteds[insti]) {
            for (int i = 0; i < m_nBeamSize; i++) {
                batchFeatureVectors.push_back(FeatureVector ());
            }
            continue;
        }

        std::vector<FeatureVector> featureVectors;
        Beam &beam = *(m_lBeamPtrs[insti].get());
        State *state = m_lLattice_indexPtrs[insti][nRound - 1];
        Instance *inst = m_lInstPtrss[insti];

        int curBeamSize = nRound == 1 ? 1 : beam.currentBeamSize;
        featureVectors.resize(curBeamSize);
        generateBatchInputForState(state, inst, featureVectors);
        for (int i = 0; i < curBeamSize; i++) {
            batchFeatureVectors.push_back(featureVectors[i]);
        }
        for (int i = curBeamSize; i < m_nBeamSize; i++) {
            batchFeatureVectors.push_back(FeatureVector ());
        }

        m_featEmbManagerPtr->returnInput(featureVectors, tnnet.modelParas->featEmbs, input, input_index);
    }
}

void BatchBeamDecoder::generateBeams(const TensorContainer<cpu, 2, real_t> &pred, const int nRound, std::vector<GlobalExample *> &gExamplePtrs, std::vector<bool> &itemCompeleteds) {
    int pred_base_index = 0;
    for (int insti = 0; insti < m_nInstSize; insti++, pred_base_index += m_nBeamSize) {
        if (itemCompeleteds[insti]) {
            continue;
        }

        Beam &beam = *(m_lBeamPtrs[insti].get());
        GlobalExample *gExample = gExamplePtrs[insti];

        beam.clear();

        State *start_state = m_lLattice_indexPtrs[insti][nRound - 1];
        State *end_state   = m_lLattice_indexPtrs[insti][nRound];
        int stateIdx = 0;
        for (State *currentState = start_state; currentState != end_state; ++currentState, ++stateIdx) {
            std::vector<int> validActs;
            m_transSystemPtr->generateValidActs(*currentState, validActs);

            bool noValid = true;
            for (unsigned actId = 0; actId < static_cast<int>(validActs.size()); ++actId) {
                if (validActs[actId] == -1) {
                    continue;
                }

                noValid = false;
                CScoredTransition trans(currentState, actId, currentState->score + pred[pred_base_index + stateIdx][actId]);

                // FOR CHECK
                trans.bGold = true;
                m_lGoldScoredTrans[insti] = trans;

                if (currentState->bGold && actId == gExample->goldActs[nRound - 1]) {
                    trans.bGold = true;
                    m_lGoldScoredTrans[insti] = trans;
                }

                beam.insert(trans);
            }
            assert (noValid == false);
        }

        // FOR CHECK
        m_lbEarlyUpdates[insti] = false;
        // m_lbEarlyUpdates[insti] = true;
        // for (int beami = 0; beami < beam.currentBeamSize; beami++) {
        //     if (beam.beam[beami].bGold) {
        //         m_lbEarlyUpdates[insti] = false;
        //     }
        // }

        if (m_lbEarlyUpdates[insti]) {
            itemCompeleteds[insti] = true;
        }
    }
}

void BatchBeamDecoder::lazyExpandBeams(const int nRound, const std::vector<bool> &itemCompeleteds, std::vector<State *> &retvals) {
    for (int insti = 0; insti < m_nInstSize; insti++) {
        if (itemCompeleteds[insti]) {
            continue;
        }

        real_t dMaxScore = 0.0;

        Beam &beam = *(m_lBeamPtrs[insti].get());
        State **lattice_index = m_lLattice_indexPtrs[insti];
        int &nGoldTransitionIndex = m_lnGoldTransitionIndex[insti];
        State *&retval = retvals[insti];

        for (int beami = 0; beami < beam.currentBeamSize; beami++) {
            const CScoredTransition &transition = beam.beam[beami];

            State *target = lattice_index[nRound] + beami;
            *target = *(transition.source);
            m_transSystemPtr->move(*(transition.source), *target, transition);

            target->bGold = transition.bGold;
            target->setBeamIdx(beami);
            if (transition.bGold) {
                nGoldTransitionIndex = beami;
            }

            if (beami == 0 || target->score > dMaxScore) {
                dMaxScore = target->score;
                retval = target;
            }
        }
    }
}

void BatchBeamDecoder::generateBatchInputForBeam(std::vector<State *> &statePtrs, std::vector<Instance *> &instPtrs, std::vector<std::vector<FeatureVector>> &featVecVecs) {
    for (int insti = 0; insti < static_cast<int>(featVecVecs.size()); insti++) {
        std::vector<FeatureVector> &featVecs = featVecVecs[insti];
        State *state = statePtrs[insti];
        Instance *inst = instPtrs[insti];

        for (int i = 0; i < static_cast<int>(featVecs.size()); i++) {
            m_featManagerPtr->extractFeature(*(state + i), *inst, featVecs[i]);
        }
    }
}

void BatchBeamDecoder::generateBatchInputForState(State *state, Instance *inst, std::vector<FeatureVector> &featVecs) {
    for (int i = 0; i < static_cast<int>(featVecs.size()); i++) {
        m_featManagerPtr->extractFeature(*(state + i), *inst, featVecs[i]);
    }
}
