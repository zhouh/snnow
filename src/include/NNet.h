// this implements a simple two layer neural net
#include <vector>
#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"
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
class NNet{
 public:
  // initialize the network
  NNet(int batch_size, int num_in, int num_hidden, int num_out) : rnd(0) {
    // setup stream
    Stream<xpu> *stream = NewStream<xpu>();
    ninput.set_stream(stream);
    nhidden.set_stream(stream);
    nhiddenbak.set_stream(stream);
    nout.set_stream(stream);
    hbias.set_stream(stream);
    g_hbias.set_stream(stream);
    
    // setup nodes
    ninput.Resize(Shape2(batch_size, num_in));
    nhidden.Resize(Shape2(batch_size, num_hidden));
    nhiddenbak.Resize(nhidden.shape_);
    nout.Resize(Shape2(batch_size, num_out));
    // setup bias
    hbias.Resize(Shape1(num_hidden)); g_hbias.Resize(hbias.shape_);
    hbias = 0.0f; obias = 0.0f;
    // setup weights
  }

  static void init(int batch_size, int num_in, int num_hidden, int num_out){
    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    g_Wi2h.set_stream(stream);
    g_Wh2o.set_stream(stream);
    Wi2h.Resize(Shape2(num_in, num_hidden));  g_Wi2h.Resize(Wi2h.shape_);
    Wh2o.Resize(Shape2(num_hidden, num_out)); g_Wh2o.Resize(Wh2o.shape_);
    rnd.SampleGaussian(&Wi2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);
  }

 ~NNet() {}
  // forward propagation
 void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                       Tensor<cpu, 2, real_t> &oubatch) {
    // size is same conventsion as numpy
    index_t batch_size = inbatch.size(0);
    // copy data to input layer
    Copy(ninput, inbatch, ninput.stream_);
    // first layer, fullc
    nhidden = dot(ninput, Wi2h);
    nhidden+= repmat(hbias, batch_size);
    // activation, sigmloid, backup activation in nhidden
    nhidden = F<sigmoid>(nhidden);
    Copy(nhiddenbak, nhidden, nhiddenbak.stream_);
    // second layer fullc
    nout = dot(nhiddenbak, Wh2o);
    // softmax calculation
    Softmax(nout, nout);
    // copy result out
    Copy(oubatch, nout, nout.stream_);
  }
  // back propagation
  void Backprop(const Tensor<cpu, 2, real_t>& gradout) {
    // copy gradient to output layer
    Copy(nout, gradout, nout.stream_);
    // calc grad of layer 2
    g_obias = sum_rows(nout);
    g_Wh2o  = dot(nhiddenbak.T(), nout);
    // backprop to layer 1
    nhiddenbak = dot(nout, Wh2o.T());
    // calculate gradient of sigmoid layer
    nhidden = nhidden * (1.0f-nhidden) * nhiddenbak;
    // calc grad of layer 1
    g_hbias = sum_rows(nhidden);
    g_Wi2h  = dot(ninput.T(), nhidden);
  }
  // update weight
  void Update(void) {
    // run SGD
    const float eta = 0.8;
    const float wd = 0.00001;
    // update weight
    Wi2h -= eta * (wd * Wi2h + g_Wi2h);
    Wh2o -= eta * (wd * Wh2o + g_Wh2o);
    // no regularization for bias
    hbias-= eta * g_hbias;
  }
 private:
  // random seed generator
  static Random<xpu, real_t> rnd;
  // static neural parameters
  TensorContainer<xpu, 2, real_t> Wi2h, Wh2o;
  TensorContainer<xpu, 1, real_t> hbias;
  // nodes in neural net
  TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
  // hidden bias, gradient
  TensorContainer<xpu, 1, real_t> g_hbias;
  // weight gradient
  TensorContainer<xpu, 2, real_t> g_Wi2h, g_Wh2o;
};
// helper function to get the max inde
inline int MaxIndex(Tensor<cpu, 1, real_t> pred) {
  int maxidx = 0;
  for(index_t i = 1; i < pred.size(0); ++i) {
    if(pred[i] > pred[maxidx]) maxidx = (int)i;
  }
  return maxidx;
}
