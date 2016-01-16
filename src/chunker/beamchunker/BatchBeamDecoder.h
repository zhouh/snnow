/*************************************************************************
	> File Name: BatchBeamDecoder.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 15 Jan 2016 02:07:47 PM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMCHUNKER_BATCHBEAMDECODER_H_
#define _CHUNKER_BEAMCHUNKER_BATCHBEAMDECODER_H_

#include <limits.h>
#include <memory>
#include <assert.h>
#include <vector>

#include "chunker.h"

#include "Beam.h"
#include "State.h"
#include "Instance.h"
#include "Example.h"
#include "LabeledSequence.h"
#include "ActionStandardSystem.h"
#include "FeatureManager.h"
#include "FeatureEmbedding.h"
#include "FeatureEmbeddingManager.h"
#include "BeamExample.h"

class TNNets;

class BatchBeamDecoderMemoryManager {
private:
    std::vector<std::vector<State *>> lattices;
    std::vector<std::vector<State **>> lattice_indexes;
    int m_nThread;
    int m_nBeamSize;
    int m_nDecoderItemSize;
public:
    BatchBeamDecoderMemoryManager(const int beamSize, const int decoderItemSize, const int longestLen, const int threadNum);
    ~BatchBeamDecoderMemoryManager();
    std::vector<State *> getLatticePtrVec(const int threadId) {
        return lattices[threadId];
    }
    
    std::vector<State **> getLatticeIndexPtrVec(const int threadId) {
        return lattice_indexes[threadId];
    }
private:
    BatchBeamDecoderMemoryManager(const BatchBeamDecoderMemoryManager &memManager) = delete;
    BatchBeamDecoderMemoryManager& operator= (const BatchBeamDecoderMemoryManager &memManager) = delete;
};

class BatchBeamDecoder {
public:
    std::shared_ptr<ActionStandardSystem> m_transSystemPtr;
    std::shared_ptr<FeatureManager> m_featManagerPtr;
    std::shared_ptr<FeatureEmbeddingManager> m_featEmbManagerPtr;

    bool m_bTrain;

    std::vector<bool> m_lbEarlyUpdates;
    std::vector<std::shared_ptr<Beam>> m_lBeamPtrs;
    std::vector<State *> m_lLatticePtrs;
    std::vector<State **> m_lLattice_indexPtrs;

    std::vector<CScoredTransition> m_lGoldScoredTrans;

    std::vector<int> m_lnGoldTransitionIndex;
    std::vector<int> m_lnMaxLatticeSizes;
    std::vector<int> m_lnRounds;
    std::vector<int> m_lnMaxRounds;
    std::vector<int> m_lnSentLens;
    std::vector<int> m_lnExpandRounds;

    std::vector<Instance *> m_lInstPtrss;
    int m_nInstSize;
    int m_nBeamSize;
public:
    BatchBeamDecoder(const std::vector<Instance *> &instancePtrs,
        const std::shared_ptr<ActionStandardSystem> &transitionSystemPtr,
        const std::shared_ptr<FeatureManager> &featureManagerPtr,
        const std::shared_ptr<FeatureEmbeddingManager> &featureEmbManagerPtr,
        int beamSize, 
        const std::vector<State *> &latticePtrs,
        const std::vector<State **> &lattice_indexPtrs,
        bool bTrain
        ) :
        m_transSystemPtr(transitionSystemPtr),
        m_featManagerPtr(featureManagerPtr),
        m_featEmbManagerPtr(featureEmbManagerPtr),
        m_bTrain(bTrain),
        m_lbEarlyUpdates(instancePtrs.size(), false),
        m_lBeamPtrs(instancePtrs.size()),
        m_lLatticePtrs(latticePtrs),
        m_lLattice_indexPtrs(lattice_indexPtrs),
        m_lGoldScoredTrans(instancePtrs.size()),
        m_lnGoldTransitionIndex(instancePtrs.size()),
        m_lnMaxLatticeSizes(instancePtrs.size()),
        m_lnRounds(instancePtrs.size(), 0),
        m_lnMaxRounds(instancePtrs.size()),
        m_lnSentLens(instancePtrs.size()),
        m_lInstPtrss(instancePtrs),
        m_nInstSize(instancePtrs.size()),
        m_nBeamSize(beamSize),
        m_lnExpandRounds(instancePtrs.size())
    {
        for (int i = 0; i < m_nInstSize; i++) {
            m_lnSentLens[i] = static_cast<int>(m_lInstPtrss[i]->input.size());
            m_lnMaxRounds[i] = m_lnSentLens[i];

            m_lnMaxLatticeSizes[i] = (beamSize + 1) * m_lnMaxRounds[i];
        }

        for (int i = 0; i < m_nInstSize; i++) {
            m_lBeamPtrs[i].reset(new Beam(m_nBeamSize));
        }
    }
    ~BatchBeamDecoder() {}

    std::vector<State *> decode(TNNets &tnnet, std::vector<GlobalExample *> &gExamplePtrs) ;
    friend class TNNets;
private:
    void generateBatchInput(const int num_in, const int nRound, const TNNets &tnnet, const std::vector<bool> &itemCompeleteds, TensorContainer<cpu, 2, real_t> &input, std::vector<FeatureVector> &batchFeatureVectors);

    void generateBeams(const TensorContainer<cpu, 2, real_t> &pred, const int nRound, std::vector<GlobalExample *> &gExamplePtrs, std::vector<bool> &itemCompeleteds);

    void lazyExpandBeams(const int nRound, const std::vector<bool> &itemCompeleteds, std::vector<State *> &retvals);

    void generateBatchInputForBeam(std::vector<State *> &statePtrs, std::vector<Instance *> &instPtrs, std::vector<std::vector<FeatureVector>> &featVecVecs);

    void generateBatchInputForState(State *state, Instance *inst, std::vector<FeatureVector> &featVecs);
};

#endif
