/*************************************************************************
	> File Name: src/include/TNNets.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 19/09/15 14:23:41
 ************************************************************************/

#ifndef SNNOW_TNNET_H
#define SNNOW_TNNET_H

#include "NNet.h"

/**
 * This is a neural net class for sequnece transition system, 
 * with which we could construct a neural net at each step of the transition
 * system, and update them toghther.
 */
class TNNets{
    
    std::vector< NNet* > nets;
    NNetParas<gpu> *netPara;
    int batch_size;
    int num_in;
    int num_hidden;
    int num_out;

    TNNets( int batch_size, int num_in, int num_hidden, int num_out , NNetParas<gpu> *para){
        this->batch_size = batch_size;
        this->num_in = num_in;
        this->num_hidden = num_hidden;
        this->num_out = num_out;
        netPara = para;
    }

    ~TNNets(){
        for( NNet* p_net : nets )
            delete p_net;
    }

    void genNextStepNet(){
        NNet<gpu> *net = new NNet<gpu>(beamSize, num_in, num_hidden, num_out, &netsParas);
        nets.push_back(net);
    }

    /*
     * computes the gradients of beam contrastive learning
     */
    void updateTNNetParas(UpdateGrads<gpu>& cumulatedGrads, Beam & beam, bool earlyUpdate, int goldTransitIndex,  CScoredTransition & goldTransit){
       
       std::vector<float> updateParas(trainingDataSize, 0); // updating parameter vector
       float sum =0;
       float maxScore = beam.getMaxScore();

       /*
        * construct the training data
        */
       std::vector<CScoredTransition*> trainingData();
       for(int bi = 0; bi < beam.currentBeamSize; bi++)
           trainingData.push_back( beam.beam + bi );
       if( earlyUpdate ){
           trainingData.push_back( & goldTransit );
           goldTransit = trainingData.size() - 1;
       }

       /*
        * get gradients with beam contrastive learning
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
       for(int backRound = nets.size() - 1; backRound >= 0; --backRound){
           TensorContainer<cpu, 2> grads;
           int i = 0;
           for(auto iter = trainingData.begin(); iter != trainingData.end(); iter++, i++){
               grads[ ( *iter )->source->beamIdx ][ ( *iter )->action ] = updateParas[i];
               *iter->source = ( *iter )->source->previous_;
               *iter->action = ( *iter )->source->last_action;
           }
           nets[backRound]->Backprop(grads);
           nets[backRound]->SubsideGrads(cumulatedGrads);
       }
    }

};

#endif
