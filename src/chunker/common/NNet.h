#ifndef SNNOW_NNET_H
#define SNNOW_NNET_H

// this implements a simple two layer neural net
#include <vector>
#include <cmath>
#include <sstream>

#include "chunker.h"

#include "Model.h"
#include "FeatureVector.h"

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// define sigmoid operation
struct sigmoid{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return  1.0f/(1.0f+expf(-a));
  }
};

struct cube{
    MSHADOW_XINLINE static real_t Map(real_t a){
        return a * a * a;
    }
};

struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

template<typename xpu>
class NNet{
public:
    // initialize the network
    NNet(int batch_size, int num_in, int num_hidden, int num_out, Model<xpu>* paras) {
        // setup stream
        this->paras = paras;

        ninput.set_stream(paras->stream);
        nhidden.set_stream(paras->stream);
        nhiddenbak.set_stream(paras->stream);
        nout.set_stream(paras->stream);
        g_hbias.set_stream(paras->stream);
        g_Wi2h.set_stream(paras->stream);
        g_Wh2o.set_stream(paras->stream);
        g_input.set_stream(paras->stream);
  
        g_Wh2o.Resize(paras->Wh2o.shape_);
        g_Wi2h.Resize(paras->Wi2h.shape_);
        g_hbias.Resize(paras->hbias.shape_);
        g_input.Resize(Shape2(batch_size, num_in));
        cpu_g_input.Resize(g_input.shape_);
        // setup nodes
        ninput.Resize(Shape2(batch_size, num_in));
        nhidden.Resize(Shape2(batch_size, num_hidden));
        nhiddenbak.Resize(nhidden.shape_);
        nout.Resize(Shape2(batch_size, num_out));
    }
  
    ~NNet() {}
    // forward propagation
    void Forward(const Tensor<cpu, 2, real_t>& inbatch,
           Tensor<cpu, 2, real_t> &oubatch, bool bDropOut){
        // size is same conventsion as numpy
        index_t batch_size = inbatch.size(0);
 
        // copy data to input layer
        Copy(ninput, inbatch, ninput.stream_);
  
        // first layer, fullc
        nhidden = dot(ninput, paras->Wi2h);
  
        nhidden+= repmat(paras->hbias, batch_size);
        // activation, sigmloid, backup activation in nhidden

        // TODO
        // nhidden = F<cube>(nhidden);
        nhidden = F<cube>(nhidden);
  
        if(bDropOut){
            TensorContainer<xpu,2, real_t> mask;
            mask.set_stream(paras->stream);
            mask.Resize(nhidden.shape_);
            paras->rnd.SampleUniform(&mask, 0.0, 1.0);
            F<threshold>(mask, CConfig::fDropoutProb);
            nhidden = nhidden * mask;
        } //dropout
  
        Copy(nhiddenbak, nhidden, nhiddenbak.stream_);
        // second layer fullc
        nout = dot(nhiddenbak, paras->Wh2o);
        // softmax calculation
      //  Softmax(nout, nout); // in TNN training, we do not need softmax in each step
        // copy result out
        Copy(oubatch, nout, nout.stream_);
    }
  
    void display1Tensor( Tensor<xpu, 1, real_t> & tensor ){
       for(int i = 0; i < tensor.size(0); i++)
           std::cout<<tensor[i]<<" ";
       std::cout<<std::endl;
    }
  
    void display2Tensor( Tensor<xpu, 2, double> tensor ){
       std::cout<<"size 0 :" << tensor.size(0)<<" size 1: "<<tensor.size(1)<<std::endl;
       for(int i = 0; i < tensor.size(0); i++){
          for(int j = 0; j < tensor.size(1); j++)
              std::cout<<tensor[i][j]<<" ";
          std::cout<<std::endl;
       }
    }
  
    void display2TensorGPU( Tensor<gpu, 2, real_t> & tensor ){
       for(int i = 0; i < tensor.size(0); i++) {
          for(int j = 0; j < tensor.size(1); j++)
              std::cout<<tensor[i][j]<<" ";
          std::cout<<std::endl;
        }
    }

    // back propagation
    void Backprop(const Tensor<cpu, 2, real_t>& gradout){
        // copy gradient to output layer
        Copy(nout, gradout, nout.stream_);
        // calc grad of layer 2
        g_Wh2o  = dot(nhiddenbak.T(), nout);
        // backprop to layer 1
        nhiddenbak = dot(nout, paras->Wh2o.T());
        //// calculate gradient of sigmoid layer
        //nhidden = nhidden * (1.0f-nhidden) * nhiddenbak;
        // calculate gradient of cube layer
        nhidden = 3 * nhidden * nhidden * nhiddenbak;
        // calc grad of layer 1
        g_hbias = sum_rows(nhidden);
        g_Wi2h  = dot(ninput.T(), nhidden);
  
        g_input = dot(nhidden, paras->Wi2h.T());
        Copy(cpu_g_input, g_input, g_input.stream_);
    }
 
    // TODO right? 
    void SubsideGradsTo(Model<xpu> *cumulatedGradsPtr, std::vector<FeatureVector> &fvs) {
        cumulatedGradsPtr->Wi2h  += g_Wi2h;
        cumulatedGradsPtr->Wh2o  += g_Wh2o;
        cumulatedGradsPtr->hbias += g_hbias;

        for (int i = 0; i < static_cast<int>(fvs.size()); i++) {
            FeatureVector &fv = fvs[i];

            int updateIndex = 0;
            for (int j = 0; j < static_cast<int>(fv.size()); j++) {
                FeatureType &ft = cumulatedGradsPtr->featTypes[j];
                std::shared_ptr<FeatureEmbedding> &curFeatEmbPtr = cumulatedGradsPtr->featEmbs[j];
                auto &oneFeatTypeVector = fv.features[j];


                for (auto &featId : oneFeatTypeVector) {
                    for (int dimi = 0; dimi < ft.featEmbSize; dimi++) {
                        curFeatEmbPtr->data[featId][dimi] += cpu_g_input[i][updateIndex++];
                    }
                }

                // for (auto &featId : oneFeatTypeVector) {
                //     curFeatEmbPtr->data[featId] += g_input[i][updateIndex];
                //     updateIndex += ft.featEmbSize;
                // }
                // for (int r = 0; r < static_cast<int>(oneFeatTypeVector.size()); r++) {
                //     cumulatedGradsPtr->featEmbs[j]->data[oneFeatTypeVector[r]] += g_input[i][updateIndex];
                //     updateIndex += ft.featEmbSize;
                // }
            }
        }
    }
  
private:
    // neural network parameters
    Model<xpu>* paras;
  
    // nodes in neural net
    TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
    // hidden bias, gradient
    TensorContainer<xpu, 1, real_t> g_hbias;
    // weight gradient
    TensorContainer<xpu, 2, real_t> g_Wi2h, g_Wh2o, g_input;

    TensorContainer<cpu, 2, real_t> cpu_g_input;
};
// template<typename xpu>
// Random<xpu, real_t> NNet<xpu>::rnd(0);

#endif
