// this implements a simple two layer neural net
#include <vector>
#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"

typedef float real_t;
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

template<typename xpu>
struct NNetPara{
  Stream<xpu> *stream = NewStream<xpu>();
  TensorContainer<xpu, 2, real_t> Wi2h, Wh2o;
  TensorContainer<xpu, 1, real_t> hbias;
  // random seed generator
  Random<xpu, real_t> rnd;  

  NNetPara(int batch_size, int num_in, int num_hidden, int num_out) : rnd(0) {

    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    hbias.set_stream(stream);
    Wi2h.Resize(Shape2(num_in, num_hidden));  
    Wh2o.Resize(Shape2(num_hidden, num_out)); 
    
    rnd.SampleGaussian(&Wi2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);

    hbias.Resize(Shape1(num_hidden)); 
    hbias = 0.0f; 
            
  }
};

template<typename xpu>
struct UpdateGrads{
    TensorContainer<xpu, 1, real_t> cg_hbias;
    TensorContainer<xpu, 2, real_t> cg_Wi2h, cg_Wh2o;

    UpdateGrads(Stream<xpu> *stream){
        cg_hbias.set_stream(stream);
        cg_Wi2h.set_stream(stream);
        cg_Wh2o.set_stream(stream);
    }

};
template<typename xpu>
class NNet{
 public:
  // initialize the network
  NNet(int batch_size, int num_in, int num_hidden, int num_out, NNetPara<xpu>* paras) {
    // setup stream
    
    this->paras = paras;
    ninput.set_stream(paras->stream);
    nhidden.set_stream(paras->stream);
    nhiddenbak.set_stream(paras->stream);
    nout.set_stream(paras->stream);
    g_hbias.set_stream(paras->stream);
    g_Wi2h.set_stream(paras->stream);
    g_Wh2o.set_stream(paras->stream);

    g_Wh2o.Resize(paras->Wh2o.shape_);
    g_Wi2h.Resize(paras->Wi2h.shape_);
    g_hbias.Resize(paras->hbias.shape_);
    // setup nodes
    ninput.Resize(Shape2(batch_size, num_in));
    nhidden.Resize(Shape2(batch_size, num_hidden));
    nhiddenbak.Resize(nhidden.shape_);
    nout.Resize(Shape2(batch_size, num_out));
  }

 ~NNet() {}
  // forward propagation
 void Forward(const Tensor<cpu, 2, real_t>& inbatch,
         Tensor<cpu, 2, real_t> &oubatch){
    // size is same conventsion as numpy
    index_t batch_size = inbatch.size(0);
    // copy data to input layer
    Copy(ninput, inbatch, ninput.stream_);
    // first layer, fullc
    nhidden = dot(ninput, paras->Wi2h);
    nhidden+= repmat(paras->hbias, batch_size);
    // activation, sigmloid, backup activation in nhidden
    nhidden = F<sigmoid>(nhidden);
    Copy(nhiddenbak, nhidden, nhiddenbak.stream_);
    // second layer fullc
    nout = dot(nhiddenbak, paras->Wh2o);
    // softmax calculation
    Softmax(nout, nout);
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
    // calculate gradient of sigmoid layer
    nhidden = nhidden * (1.0f-nhidden) * nhiddenbak;
    // calc grad of layer 1
    g_hbias = sum_rows(nhidden);
    g_Wi2h  = dot(ninput.T(), nhidden);
 }

  // update weight
 void Update(void){
    // run SGD
    const float eta = 0.8;
    const float wd = 0.00001;
    // update weight
    paras->Wi2h -= eta * (wd * paras->Wi2h + g_Wi2h);
    paras->Wh2o -= eta * (wd * paras->Wh2o + g_Wh2o);
    // no regularization for bias
    paras->hbias-= eta * g_hbias;
 }
 void SubsideGrads(UpdateGrads<xpu>& cumulatedGrads){
    cumulatedGrads.cg_hbias += cumulatedGrads.cg_hbias + g_hbias;
    cumulatedGrads.cg_Wi2h += cumulatedGrads.cg_Wi2h + g_Wi2h;
    cumulatedGrads.cg_Wh2o += cumulatedGrads.cg_Wh2o + g_Wh2o;    
 }
 static void UpdateCumulateGrads(UpdateGrads<xpu>& cgrads, NNetPara<xpu>* paras){
    // run SGD
    const float eta = 0.8;
    const float wd = 0.00001;
    // update weight
    paras->Wi2h -= eta * (wd * paras->Wi2h + cgrads.cg_Wi2h);
    paras->Wh2o -= eta * (wd * paras->Wh2o + cgrads.cg_Wh2o);
    // no regularization for bias
    paras->hbias-= eta * cgrads.cg_hbias;

 }

 private:

 // neural network parameters
 NNetPara<xpu>* paras;

  // nodes in neural net
  TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
  // hidden bias, gradient
  TensorContainer<xpu, 1, real_t> g_hbias;
  // weight gradient
  TensorContainer<xpu, 2, real_t> g_Wi2h, g_Wh2o;
};

