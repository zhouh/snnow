// this implements a simple two layer neural net
#include<NNet.h>

template<typename xpu>
void NNet<xpu>::init(int batch_size, int num_in, int num_hidden, int num_out){
    // setup stream
    Stream<xpu> *stream = NewStream<xpu>();
    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    g_Wi2h.set_stream(stream);
    g_Wh2o.set_stream(stream);
    Wi2h.Resize(Shape2(num_in, num_hidden));  g_Wi2h.Resize(Wi2h.shape_);
    Wh2o.Resize(Shape2(num_hidden, num_out)); g_Wh2o.Resize(Wh2o.shape_);
    
    Random<xpu, real_t> rnd(0);
    rnd.SampleGaussian(&Wi2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);
}

// forward propagation
template<typename xpu>
void NNet<xpu>::Forward(const Tensor<cpu, 2, real_t>& inbatch,
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
template<typename xpu>
void NNet<xpu>::Backprop(const Tensor<cpu, 2, real_t>& gradout) {
    // copy gradient to output layer
    Copy(nout, gradout, nout.stream_);
    // calc grad of layer 2
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
template<typename xpu>
void NNet<xpu>::Update(void) {
    // run SGD
    const float eta = 0.8;
    const float wd = 0.00001;
    // update weight
    Wi2h -= eta * (wd * Wi2h + g_Wi2h);
    Wh2o -= eta * (wd * Wh2o + g_Wh2o);
    // no regularization for bias
    hbias-= eta * g_hbias;
}
