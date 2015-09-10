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
class NNet{
 public:
  // initialize the network
  NNet(int batch_size, int num_in, int num_hidden, int num_out) {
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
    hbias = 0.0f; 
    // setup weights
  }

 ~NNet() {}
  // forward propagation
 void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                       Tensor<cpu, 2, real_t> &oubatch);
  // back propagation
  void Backprop(const Tensor<cpu, 2, real_t>& gradout);

  // init the static members of the network!
  static void init(int batch_size, int num_in, int num_hidden, int num_out);
  // update weight
 static void Update(void); 

 private:
  // random seed generator
  static Random<xpu, real_t> rnd;
  // static neural parameters
  static TensorContainer<xpu, 2, real_t> Wi2h, Wh2o;
  static TensorContainer<xpu, 1, real_t> hbias;
  // nodes in neural net
  TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
  // hidden bias, gradient
  static  TensorContainer<xpu, 1, real_t> g_hbias;
  // weight gradient
  static  TensorContainer<xpu, 2, real_t> g_Wi2h, g_Wh2o;
};
