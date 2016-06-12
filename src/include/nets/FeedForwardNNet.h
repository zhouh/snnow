//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_FEEDFORWARDNNET_H
#define SNNOW_FEEDFORWARDNNET_H

// this implements a simple two layer neural net
#include <vector>
#include <cmath>
#include <sstream>


#include "Model.h"
#include "FeatureVector.h"
#include "Activation.h"

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

DECLARE_double(dropout_prob);

template<typename xpu>
class FeedForwardNNet{

public:
    // initialize the network
    FeedForwardNNet(const int batch_size, const int input_num, const int hidden_num, const int output_num, Model<cpu>* model_paras) {
        this->paras = paras;

        // setup stream
        // stream = NewStream<xpu>();
        stream = paras->stream;

        g_hbias.set_stream(stream);
        g_Wi2h.set_stream(stream);
        g_Wh2o.set_stream(stream);
        g_input.set_stream(stream);

        g_Wh2o.Resize(paras->Wh2o.shape_, static_cast<real_t>(0.0));
        g_Wi2h.Resize(paras->Wi2h.shape_, static_cast<real_t>(0.0));
        g_hbias.Resize(paras->hbias.shape_, static_cast<real_t>(0.0));
        g_input.Resize(Shape2(batch_size, input_num), static_cast<real_t>(0.0));
        XPU_g_input.Resize(g_input.shape_, static_cast<real_t>(0.0));

        ninput.set_stream(stream);
        nhidden.set_stream(stream);
        nhiddenbak.set_stream(stream);
        nout.set_stream(stream);

        // setup nodes
        ninput.Resize(Shape2(batch_size, input_num), static_cast<real_t>(0.0));
        nhidden.Resize(Shape2(batch_size, hidden_num), static_cast<real_t>(0.0));
        nhiddenbak.Resize(nhidden.shape_, static_cast<real_t>(0.0));
        nout.Resize(Shape2(batch_size, output_num), static_cast<real_t>(0.0));
    }

    // ~NNet() { DeleteStream(stream); }
    ~FeedForwardNNet() {  }

    // forward propagation
    void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                 Tensor<cpu, 2, real_t> &oubatch, bool bDropOut){
        // size is same conventsion as numpy
        index_t batch_size = inbatch.size(0);

        // copy data to input layer
        Copy(ninput, inbatch, ninput.stream_);

        // first layer, fullc
        nhidden  = dot(ninput, paras->Wi2h);

        nhidden += repmat(paras->hbias, batch_size);

        // nhiddenbak = F<sigmoid>(nhidden);

        if(bDropOut){
            TensorContainer<xpu,2, real_t> mask;
            mask.set_stream(stream);
            mask.Resize(nhiddenbak.shape_);

            paras->rnd.SampleUniform(&mask, 0.0, 1.0);

            mask = F<threshold>(mask, FLAGS_dropout_prob);

            nhidden = nhidden * mask;
            // nhiddenbak = nhiddenbak * mask;
        } //dropout

        // activation, sigmloid, backup activation in nhidden
        nhiddenbak = F<cube>(nhidden);

        // second layer fullc
        nout = dot(nhiddenbak, paras->Wh2o);

        // softmax calculation
        //  Softmax(nout, nout); // in TNN training, we do not need softmax in each step

        // copy result out
        Copy(oubatch, nout, nout.stream_);
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
        // nhidden = nhidden * (1.0f - nhidden) * nhiddenbak;

        // calculate gradient of cube layer
        nhidden = 3 * nhidden * nhidden * nhiddenbak;

        // calc grad of layer 1
        g_hbias = sum_rows(nhidden);
        g_Wi2h  = dot(ninput.T(), nhidden);

            g_input = dot(nhidden, paras->Wi2h.T());
            Copy(XPU_g_input, g_input, g_input.stream_);
    }

    // TODO right?
    void SubsideGradsTo(Model<xpu> *cumulatedGradsPtr, std::vector<FeatureVector> &fvs) {
        cumulatedGradsPtr->Wi2h  += g_Wi2h;
        cumulatedGradsPtr->Wh2o  += g_Wh2o;
        cumulatedGradsPtr->hbias += g_hbias;

            for (int i = 0; i < static_cast<int>(fvs.size()); i++) {
                FeatureVector &fv = fvs[i];
                if (fv.size() == 0) {
                    continue;
                }

                int updateIndex = 0;
                for (int j = 0; j < static_cast<int>(fv.size()); j++) {
                    FeatureType &ft = cumulatedGradsPtr->featTypes[j];
                    std::shared_ptr<FeatureEmbedding> &curFeatEmbPtr = cumulatedGradsPtr->featEmbs[j];
                    auto &oneFeatTypeVector = fv[j];


                    for (auto &featId : oneFeatTypeVector) {
                        curFeatEmbPtr->data[featId] += XPU_g_input[i].Slice(updateIndex, updateIndex + ft.feature_embedding_size);
                        updateIndex += ft.feature_embedding_size;
                        // for (int dimi = 0; dimi < ft.featEmbSize; dimi++) {
                        //     curFeatEmbPtr->data[featId][dimi] += XPU_g_input[i][updateIndex++];
                        // }
                    }
                }
            }
    }

private:
    // neural network parameters
    Model<xpu>* paras;

    Stream<xpu> *stream;

    // nodes in neural net
    TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
    // hidden bias, gradient
    TensorContainer<xpu, 1, real_t> g_hbias;
    // weight gradient
    TensorContainer<xpu, 2, real_t> g_Wi2h, g_Wh2o, g_input;

    TensorContainer<cpu, 2, real_t> XPU_g_input;

public:
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

};


#endif //SNNOW_FEEDFORWARDNNET_H
