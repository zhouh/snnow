/*************************************************************************
	> File Name: TNNets.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 16 Jan 2016 03:07:15 PM CST
 ************************************************************************/

#include "TNNets.h"
#include "BatchBeamDecoder.h"
#include "BeamDecoder.h"

TNNetsMemoryManager::TNNetsMemoryManager(const int threadNum, const int longestLen, const int batchSize, const int num_in, const int num_hidden, const int num_out, Model<XPU> *modelParasPtr) : m_nThread(threadNum) {
    for (int i = 0; i < m_nThread; i++) {
        std::vector<NNet<XPU> *> nets(longestLen + 1);

        for (int j = 0; j < nets.size(); j++) {
            nets[j] = new NNet<XPU>(batchSize, num_in, num_hidden, num_out, modelParasPtr);
        }

        netss.push_back(nets);
    }
}

TNNetsMemoryManager::~TNNetsMemoryManager() {
    for (int i = 0; i < m_nThread; i++) {
        for (int j = 0; j < netss[i].size(); j++) {
            delete netss[i][j];
        }
    }
}

std::vector<NNet<XPU> *> TNNetsMemoryManager::getNetPtrVec(const int threadId) {
    return netss[threadId];
}

/*
 * computes the gradients of beam contrastive learning
 */
void TNNets::updateTNNetParas(Model<XPU> *cumulatedGrads, BeamDecoder &decoder) {
    Beam &beam = decoder.beam;
    bool earlyUpdate = decoder.bEarlyUpdate;
    int goldTransitIndex = decoder.nGoldTransitionIndex;
    CScoredTransition &goldTransit = decoder.goldScoredTran;
   
    float sum =0;
    float maxScore = beam.getMaxScoreInBeam();

    // TODO: predict correctly ?
    if (!earlyUpdate && beam.isMaxScoreGold()) {
        return ;
    }
    /*
     * construct the training data
     */
    std::vector<CScoredTransition*> trainingData;

    for(int bi = 0; bi < beam.currentBeamSize; bi++)
        trainingData.push_back( beam.beam + bi );
    if( earlyUpdate ){
        trainingData.push_back( & goldTransit );
        goldTransitIndex = trainingData.size() - 1;
    }

    std::vector<double> updateParas(trainingData.size(), 0); // updating parameter vector

    /*
     * get gradients with beam contrastive learning
     * sentence-level loglikelihood and softmax
     */
    for (int b_j = 0; b_j < trainingData.size(); b_j++) { // for every transit in the beam
        updateParas[b_j] = exp( trainingData[b_j]->score - maxScore );
        sum += updateParas[b_j];
    }
    for (int b_j = 0; b_j < trainingData.size(); b_j++)
        updateParas[b_j] = updateParas[b_j] / sum;
    updateParas[ goldTransitIndex ] -= 1.0;

    /*  
     *  Back propagation updating
     *  from last parsing state to the former states
     */
    for(int backRound = netIdx - 1; backRound >= 0; --backRound){
        //std::cout<<"backRound:\t"<<backRound<<std::endl;
        TensorContainer<cpu, 2, real_t> grads;
        grads.Resize( Shape2( batch_size, num_out ) );
        grads = 0.0;
        int i = 0;
        for(auto iter = trainingData.begin(); iter != trainingData.end(); iter++, i++){
            grads[ ( *iter )->source->beamIdx ][ ( *iter )->action ] += updateParas[i] / CConfig::nBeamBatchSize;
            if( backRound != 0 ){ // last time updating, do not need to prepare for next iteration
                 ( *iter )->action = ( *iter )->source->lastAction;
                 ( *iter )->source = ( *iter )->source->prevStatePtr;
            }
        }

        // FOR CHECK
        nets[backRound]->Backprop(grads);

        nets[backRound]->SubsideGradsTo(cumulatedGrads, netFeatVecs[backRound]);
    }
}

void TNNets::updateTNNetParas(Model<XPU> *cumulatedGrads, BatchBeamDecoder &batchDecoder) {
    const int itemSize = batchDecoder.m_nInstSize;
    std::vector<real_t> maxScores(itemSize, 0.0);
    std::vector<std::vector<real_t>> updateParasVec(itemSize);

    std::vector<std::vector<CScoredTransition *>> trainingDatas(itemSize);
    std::vector<bool> predictCorrect(itemSize, false);

    for (int insti = 0; insti < itemSize; insti++) {
        real_t sum = 0.0;
        real_t &maxScore = maxScores[insti];

        Beam &beam = *(batchDecoder.m_lBeamPtrs[insti].get());

        maxScore = beam.getMaxScoreInBeam();

        bool earlyUpdate = batchDecoder.m_lbEarlyUpdates[insti];

        if (!earlyUpdate && beam.isMaxScoreGold()) {
            predictCorrect[insti] = true;
        }

        int &goldTransitIndex = batchDecoder.m_lnGoldTransitionIndex[insti];
        CScoredTransition &goldTransit = batchDecoder.m_lGoldScoredTrans[insti];
        std::vector<CScoredTransition *> &trainingData = trainingDatas[insti];

        for (int beami = 0; beami < beam.currentBeamSize; beami++) {
            trainingData.push_back(beam.beam + beami);
        }
        if (earlyUpdate) {
            trainingData.push_back(&goldTransit);
            goldTransitIndex = trainingData.size() - 1;
        }

        updateParasVec[insti] = std::vector<real_t>(trainingData.size(), 0.0);

        std::vector<real_t> &updateParas = updateParasVec[insti];
        for (int beami = 0; beami < static_cast<int>(trainingData.size()); beami++) {
            updateParas[beami] = exp(trainingData[beami]->score - maxScore);
            sum += updateParas[beami];
        }
        for (int beami = 0; beami < static_cast<int>(trainingData.size()); beami++) {
            updateParas[beami] = updateParas[beami] / sum;
        }
        updateParas[goldTransitIndex] -= 1.0;
    }

    for (int backRound = netIdx - 1; backRound >= 0; --backRound) {
        TensorContainer<cpu, 2, real_t> grads(Shape2(batch_size, num_out), static_cast<real_t>(0.0));

        int baseIndex = 0;
        for (int insti = 0; insti < itemSize; insti++, baseIndex += batchDecoder.m_nBeamSize) {
            if (predictCorrect[insti] || batchDecoder.m_lnExpandRounds[insti] < backRound + 1) {
                continue;
            }
           
            std::vector<CScoredTransition *> &trainingData = trainingDatas[insti];
            std::vector<real_t> &updateParas = updateParasVec[insti];
            int i = 0;
            for (auto iter = trainingData.begin(); iter != trainingData.end(); iter++, i++) {
                grads[(*iter)->source->beamIdx + baseIndex][(*iter)->action] += updateParas[i] / CConfig::nBeamBatchSize;

                if (backRound != 0) {
                    (*iter)->action = (*iter)->source->lastAction;
                    (*iter)->source = (*iter)->source->prevStatePtr;
                }
            }
        }

        // FOR CKECK
        nets[backRound]->Backprop(grads);
        nets[backRound]->SubsideGradsTo(cumulatedGrads, netFeatVecs[backRound]);
    }
}
