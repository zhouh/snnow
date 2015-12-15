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
    
public:
    std::vector< NNet<XPU>* > nets;
    NNetPara<XPU> *netPara;
    int batch_size;
    int num_in;
    int num_hidden;
    int num_out;
    bool bTrain;

public:
    TNNets( int batch_size, int num_in, int num_hidden, int num_out , NNetPara<XPU> *para, bool bTrain = true): netPara(para){
        this->batch_size = batch_size;
        this->num_in = num_in;
        this->num_hidden = num_hidden;
        this->num_out = num_out;
        //netPara = para;
        this->bTrain = bTrain;
        if( !bTrain )
            genNextStepNet(); // in testing, we only need one neural net for forwarding
    }

    ~TNNets(){
        for( NNet<XPU>* p_net : nets )
            delete p_net;
    }

    void genNextStepNet(){
        NNet<XPU> *net = new NNet<XPU>(batch_size, num_in, num_hidden, num_out, netPara);
        nets.push_back(net);
    }
    
    void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                  Tensor<cpu, 2, real_t> &oubatch){
        nets[nets.size() - 1]->Forward(inbatch, oubatch, bTrain && CConfig::bDropOut);
    }

    /*
     * computes the gradients of beam contrastive learning
     */
    void updateTNNetParas(UpdateGrads<XPU> & cumulatedGrads, Beam & beam, bool earlyUpdate, int goldTransitIndex,  CScoredTransition & goldTransit){
       
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
       for(int backRound = nets.size() - 1; backRound >= 0; --backRound){
           //std::cout<<"backRound:\t"<<backRound<<std::endl;
           TensorContainer<cpu, 2, real_t> grads;
           grads.Resize( Shape2( batch_size, num_out ) );
           grads = 0.0;
           int i = 0;
           for(auto iter = trainingData.begin(); iter != trainingData.end(); iter++, i++){
               //( *iter )->source->printActionSequence();
               //std::cout<<"action :\t"<<(*iter)->action<<std::endl;
               grads[ ( *iter )->source->beamIdx ][ ( *iter )->action ] += updateParas[i] / CConfig::nBatchSize;
               if( backRound != 0 ){ // last time updating, do not need to prepare for next iteration
                    ( *iter )->action = ( *iter )->source->last_action;
                    ( *iter )->source = ( *iter )->source->previous_;
               }
           }
           for (int ii = 0; ii < batch_size; ii++) {
               for (int jj = 0; jj < num_out; jj++) {
                   if (isnan(grads[ii][jj])) {
                       std::cout << "[before backprop]: nan appears in grads" << std::endl;
                   }
               }
           }

           //std::cout<<"begin to back propagation!"<<std::endl;
           nets[backRound]->Backprop(grads);

           for (int ii = 0; ii < nets[backRound]->g_Wi2h.shape_[0]; ii++) {
               for (int jj = 0; jj < nets[backRound]->g_Wi2h.shape_[1]; jj++){
                   if (isnan(nets[backRound]->g_Wi2h[ii][jj])) {
                       std::cout << "[after backprop]W(input -> hidden): NaN appears!" << std::endl;
                   }
               }
           }
           for (int ii = 0; ii < nets[backRound]->g_Wh2o.shape_[0]; ii++) {
               for (int jj = 0; jj < nets[backRound]->g_Wh2o.shape_[1]; jj++){
                   if (isnan(nets[backRound]->g_Wh2o[ii][jj])) {
                       std::cout << "[after backprop]W(hidden -> output): NaN appears!" << std::endl;
                   }
               }
           }
           for (int ii = 0; ii < nets[backRound]->g_hbias.shape_[0]; ii++) {
               if (isnan(nets[backRound]->g_hbias[ii])) {
                   std::cout << "[after backprop]Bias: NaN appears!" << std::endl;
               }
           }
           //std::cout<<"begin to back subsidegrads!"<<std::endl;
           nets[backRound]->SubsideGrads(cumulatedGrads);
           for (int ii = 0; ii < cumulatedGrads.cg_Wi2h.shape_[0]; ii++) {
               for (int jj = 0; jj < cumulatedGrads.cg_Wi2h.shape_[1]; jj++){
                   if (isnan(cumulatedGrads.cg_Wi2h[ii][jj])) {
                       std::cout << "W(input -> hidden): NaN appears!" << std::endl;
                   }
               }
           }
           for (int ii = 0; ii < cumulatedGrads.cg_Wh2o.shape_[0]; ii++) {
               for (int jj = 0; jj < cumulatedGrads.cg_Wh2o.shape_[1]; jj++){
                   if (isnan(cumulatedGrads.cg_Wh2o[ii][jj])) {
                       std::cout << "W(hidden -> output): NaN appears!" << std::endl;
                   }
               }
           }
           for (int ii = 0; ii < cumulatedGrads.cg_hbias.shape_[0]; ii++) {
               if (isnan(cumulatedGrads.cg_hbias[ii])) {
                   std::cout << "Bias: NaN appears!" << std::endl;
               }
           }
       }
    }

};

#endif
