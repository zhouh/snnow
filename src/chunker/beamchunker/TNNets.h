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
    TNNets( int batch_size, int num_in, int num_hidden, int num_out , Model<XPU> *para, std::vector<NNet<XPU> *> &nnets, bool bTrain = true): nets(nnets), modelParas(para){
        assert (bTrain);

        this->batch_size = batch_size;
        this->num_in = num_in;
        this->num_hidden = num_hidden;
        this->num_out = num_out;
        this->bTrain = bTrain;
        netIdx = 0;
    }

    TNNets( int batch_size, int num_in, int num_hidden, int num_out , Model<XPU> *para, bool bTrain = true): modelParas(para){
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

    /*
     * computes the gradients of beam contrastive learning
     */
    void updateTNNetParas(Model<XPU> *cumulatedGrads, Beam & beam, bool earlyUpdate, int goldTransitIndex,  CScoredTransition & goldTransit){
       
       float sum =0;
       float maxScore = beam.getMaxScoreInBeam();

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
               //( *iter )->source->printActionSequence();
               //std::cout<<"action :\t"<<(*iter)->action<<std::endl;
               grads[ ( *iter )->source->beamIdx ][ ( *iter )->action ] += updateParas[i] / CConfig::nBeamBatchSize;
               if( backRound != 0 ){ // last time updating, do not need to prepare for next iteration
                    ( *iter )->action = ( *iter )->source->last_action;
                    ( *iter )->source = ( *iter )->source->previous_;
               }
           }

           nets[backRound]->Backprop(grads);

           nets[backRound]->SubsideGradsTo(cumulatedGrads, netFeatVecs[backRound]);
       }
    }

};

#endif
