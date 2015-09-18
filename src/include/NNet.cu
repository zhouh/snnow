// this implements a simple two layer neural net
#include<NNet.h>

// forward propagation
template<typename xpu>
void NNet<xpu>::Forward(const Tensor<cpu, 2, real_t>& inbatch,
                       Tensor<cpu, 2, real_t> &oubatch) {
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
template<typename xpu>
void NNet<xpu>::Backprop(const Tensor<cpu, 2, real_t>& gradout) {
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
template<typename xpu>
void NNet<xpu>::Update(void) {
    // run SGD
    const float eta = 0.8;
    const float wd = 0.00001;
    // update weight
    paras->Wi2h -= eta * (wd * paras->Wi2h + g_Wi2h);
    paras->Wh2o -= eta * (wd * paras->Wh2o + g_Wh2o);
    // no regularization for bias
    paras->hbias-= eta * g_hbias;
}

template<typename xpu>
void NNet<xpu>::SubsideGrads(UpdateGrads<xpu>& cumulatedGrads){
    cumulatedGrads.cg_hbias += cumulatedGrads.cg_hbias + g_hbias;
    cumulatedGrads.cg_Wi2h += cumulatedGrads.cg_Wi2h + g_Wi2h;
    cumulatedGrads.cg_Wh2o += cumulatedGrads.cg_Wh2o + g_Wh2o;
}
