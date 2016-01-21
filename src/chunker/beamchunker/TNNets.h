/*************************************************************************
	> File Name: src/include/TNNets.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 19/09/15 14:23:41
 ************************************************************************/
#ifndef _CHUNKER_BEAMCHUNKER_TNNETS_H_
#define _CHUNKER_BEAMCHUNKER_TNNETS_H_

#include <assert.h>

#include "chunker.h"

#include "NNet.h"

#include "Beam.h"
#include "State.h"
#include "FeatureVector.h"
#include "BeamExample.h"

class BatchBeamDecoder;
class BeamDecoder;

class TNNetsMemoryManager {
private:
    std::vector<std::vector<NNet<XPU> *>> netss;
    int m_nThread;

public:
    TNNetsMemoryManager(const int threadNum, const int longestLen, const int batchSize, const int num_in, const int num_hidden, const int num_out, Model<XPU> *modelParasPtr);
    ~TNNetsMemoryManager();
    std::vector<NNet<XPU> *> getNetPtrVec(const int threadId);
private:
    TNNetsMemoryManager(const TNNetsMemoryManager &memManager) = delete;
    TNNetsMemoryManager& operator= (const TNNetsMemoryManager &memManager) = delete;
};

/**
 * This is a neural net class for sequnece transition system, 
 * with which we could construct a neural net at each step of the transition
 * system, and update them toghther.
 */
class TNNets{
public:
    std::vector< NNet<XPU>* > nets;
    std::vector< std::vector<FeatureVector> > netFeatVecs;
    Model<XPU> *modelParas;
    int batch_size;
    int num_in;
    int num_hidden;
    int num_out;
    bool bTrain;

    int netIdx;

public:
    TNNets( const int batch_size, const int num_in, const int num_hidden, const int num_out , Model<XPU> *para, const std::vector<NNet<XPU> *> &nnets, bool bTrain = true): nets(nnets), modelParas(para){
        assert (bTrain);

        this->batch_size = batch_size;
        this->num_in = num_in;
        this->num_hidden = num_hidden;
        this->num_out = num_out;
        this->bTrain = bTrain;
        netIdx = 0;
    }

    TNNets( const int batch_size, const int num_in, const int num_hidden, const int num_out, Model<XPU> *para, bool bTrain = true): modelParas(para){
        assert (!bTrain);

        this->batch_size = batch_size;
        this->num_in = num_in;
        this->num_hidden = num_hidden;
        this->num_out = num_out;
        //modelParas = para;
        this->bTrain = bTrain;

        netIdx = 0;
        if( !bTrain )
            genNextStepNet(); // in testing, we only need one neural net for forwarding
    }

    ~TNNets(){
        if (!bTrain) {
            for( NNet<XPU>* p_net : nets )
                delete p_net;
        }
    }

    void moveToNextNet() {
        netIdx++;

        assert (netIdx < nets.size());
    }

    void genNextStepNet(){
        NNet<XPU> *net = new NNet<XPU>(batch_size, num_in, num_hidden, num_out, modelParas);
        nets.push_back(net);
        netIdx++;
    }
   
    void addFeatVecs(std::vector<FeatureVector> &featVecs) {
        netFeatVecs.push_back(featVecs);
    }

    void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                  Tensor<cpu, 2, real_t> &oubatch){
        nets[netIdx - 1]->Forward(inbatch, oubatch, bTrain && CConfig::bDropOut);
    }

    void updateTNNetParas(Model<XPU> *cumulatedGrads, BeamDecoder &decoder);

    void updateTNNetParas(Model<XPU> *cumulatedGrads, BatchBeamDecoder &batchDecoder);

};

#endif
