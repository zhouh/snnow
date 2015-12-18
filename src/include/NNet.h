#ifndef SNNOW_NNET_H
#define SNNOW_NNET_H

// this implements a simple two layer neural net
#include <vector>
#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"

typedef double real_t;
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

struct square{
    MSHADOW_XINLINE static real_t Map(real_t a){
        return  a * a;
    }
};

struct mySqrt{
    MSHADOW_XINLINE static real_t Map(real_t a){
        return  sqrt(a);
    }
};

/*
 * parameters of a neural net
 */
template<typename xpu>
struct NNetPara{
  Stream<xpu> *stream = NewStream<xpu>();
  TensorContainer<xpu, 2, real_t> Wi2h, Wh2o;
  TensorContainer<xpu, 1, real_t> hbias;
  TensorContainer<xpu, 2, real_t> eg2Wi2h, eg2Wh2o;
  TensorContainer<xpu, 1, real_t> eg2Hbias;
  // random seed generator
  Random<xpu, real_t> rnd;  
  

  NNetPara(int batch_size, int num_in, int num_hidden, int num_out) : rnd(0) {

    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    hbias.set_stream(stream);
    Wi2h.Resize(Shape2(num_in, num_hidden), static_cast<real_t>(0.0));  
    Wh2o.Resize(Shape2(num_hidden, num_out), static_cast<real_t>(0.0)); 
    
    rnd.SampleUniform(&Wi2h, -0.01, 0.01f);
    rnd.SampleUniform(&Wh2o, -0.01, 0.01f);

    
    hbias.Resize(Shape1(num_hidden), static_cast<real_t>(0.0)); 
    rnd.SampleUniform(&hbias, -0.01, 0.01f);

    /*
     * assert the tensor is not NaN
     */
    /*assert( !is2TensorNaN(Wi2h) );*/
    //assert( !is2TensorNaN(Wh2o) );
    //assert( !is1TensorNaN(hbias) );

    eg2Wi2h.set_stream(stream);
    eg2Wh2o.set_stream(stream);
    eg2Hbias.set_stream(stream);
    eg2Wi2h.Resize(Shape2(num_in, num_hidden), static_cast<real_t>(0.0));  
    eg2Wh2o.Resize(Shape2(num_hidden, num_out), static_cast<real_t>(0.0)); 
    eg2Hbias.Resize(Shape1(num_hidden), static_cast<real_t>(0.0)); 

  }

  void saveModel( std::ostream & os ){
      
      /*
       * write the Wi2h
       */
      os << Wi2h.size(0) << "\t" << Wi2h.size( 1 ) << std::endl;
      for( index_t i = 0; i < Wi2h.size( 0 ); i++ )
          for( index_t j = 0; j < Wi2h.size( 1 ); j++ ){
              os << Wi2h[ i ][ j ];
              if( j == ( Wi2h.size( 1 ) - 1 ) )
                  os << std::endl;
              else
                  os << " ";

          }
      
      /*
       * write the Wh2o
       */
      os << Wh2o.size(0) << "\t" << Wh2o.size( 1 ) << std::endl;
      for( index_t i = 0; i < Wh2o.size( 0 ); i++ )
          for( index_t j = 0; j < Wh2o.size( 1 ); j++ ){
              os << Wh2o[ i ][ j ];
              if( j == ( Wh2o.size( 1 ) - 1 ) )
                  os << std::endl;
              else
                  os << " ";

          }

      /*
       * write the hbias
       */
      os << hbias.size(0) << std::endl;
      for( index_t i = 0; i < hbias.size( 0 ); i++ ){
          os << hbias[ i ];
          if(  i == ( hbias.size( 0 ) - 1 ) )
              os << std::endl;
          else
              os << " ";

      }
  }

  void loadModel( std::istream & is ){
      
      std::string line;
      index_t size0, size1;
      /*
       * read Wi2h
       */
      getline( is, line );
      std::istringstream iss(line);
      iss >> size0 >> size1; 
      for( index_t i = 0; i < size0; i++  ){
          getline( is, line );
          std::istringstream iss_j( line );
          for( index_t j = 0; j < size1; j++ )
              iss_j >> Wi2h[ i ][ j ];
      }

      /*
      * read Wh2o
      */
      getline( is, line );
      std::istringstream iss_wh2o(line);
      iss_wh2o >> size0 >> size1; 
      for( index_t i = 0; i < size0; i++  ){
          getline( is, line );
          std::istringstream iss_wh2o_j( line );
          for( index_t j = 0; j < size1; j++ )
              iss_wh2o_j >> Wh2o[ i ][ j ];
      }

      /*
      * read hbias
      */
      getline( is, line );
      std::istringstream iss_hbias(line);
      iss_hbias >> size0; 
      getline( is, line );
      std::istringstream iss_hbias_j( line );
      for( index_t i = 0; i < size0; i++  ){
          iss_hbias_j >> hbias[ i ];
      }
  }

};

/*
 * used for multi-thread mini-batch training for temporary store
 * gradients of each thread
 */
template<typename xpu>
struct UpdateGrads{
    TensorContainer<xpu, 1, real_t> cg_hbias;
    TensorContainer<xpu, 2, real_t> cg_Wi2h, cg_Wh2o;

    UpdateGrads(Stream<xpu> *stream, int num_in, int num_hidden, int num_out){
        cg_hbias.set_stream(stream);
        cg_Wi2h.set_stream(stream);
        cg_Wh2o.set_stream(stream);

        cg_Wi2h.Resize(Shape2(num_in, num_hidden), static_cast<real_t>(0.0));  
        cg_Wh2o.Resize(Shape2(num_hidden, num_out), static_cast<real_t>(0.0)); 
        cg_hbias.Resize(Shape1(num_hidden), static_cast<real_t>(0.0)); 
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

    g_Wh2o.Resize(paras->Wh2o.shape_, static_cast<real_t>(0.0));
    g_Wi2h.Resize(paras->Wi2h.shape_, static_cast<real_t>(0.0));
    g_hbias.Resize(paras->hbias.shape_, static_cast<real_t>(0.0));
    // setup nodes
    ninput.Resize(Shape2(batch_size, num_in), static_cast<real_t>(0.0));
    nhidden.Resize(Shape2(batch_size, num_hidden), static_cast<real_t>(0.0));
    nhiddenbak.Resize(nhidden.shape_, static_cast<real_t>(0.0));
    nout.Resize(Shape2(batch_size, num_out), static_cast<real_t>(0.0));
  }

  NNet(NNet* net){
    this->paras = net->paras;
    ninput.set_stream(paras->stream);
    nhidden.set_stream(paras->stream);
    nhiddenbak.set_stream(paras->stream);
    nout.set_stream(paras->stream);
    g_hbias.set_stream(paras->stream);
    g_Wi2h.set_stream(paras->stream);
    g_Wh2o.set_stream(paras->stream);

    g_Wh2o.Resize(paras->Wh2o.shape_, static_cast<real_t>(0.0));
    g_Wi2h.Resize(paras->Wi2h.shape_, static_cast<real_t>(0.0));
    g_hbias.Resize(paras->hbias.shape_, static_cast<real_t>(0.0));
    // setup nodes
    ninput.Resize(net->ninput.shape_, static_cast<real_t>(0.0));
    nhidden.Resize( Shape2(net->nhidden.shape_), static_cast<real_t>(0.0) );
    nhiddenbak.Resize(net->nhiddenbak.shape_, static_cast<real_t>(0.0));
    nout.Resize(net->nout.shape_, static_cast<real_t>(0.0)); 
  }

 ~NNet() {}
  // forward propagation
 void Forward(const Tensor<cpu, 2, real_t>& inbatch,
         Tensor<cpu, 2, real_t> &oubatch, bool bDropOut){

    //display2Tensor(inbatch);
    TensorContainer<cpu, 2, double> copytensor;

    // size is same conventsion as numpy
    index_t batch_size = inbatch.size(0);
    // copy data to input layer
    Copy(ninput, inbatch, ninput.stream_);

    // first layer, fullc
    nhidden = dot(ninput, paras->Wi2h);

/*    copytensor.Resize(nhidden.shape_);*/
    //Copy(copytensor, nhidden, nhidden.stream_);
    //display2Tensor(copytensor);

    //std::cout<<"Wi2h"<<std::endl;
    //copytensor.Resize(paras->Wi2h.shape_);
    //Copy(copytensor, paras->Wi2h, paras->Wi2h.stream_);
    //display2Tensor(copytensor);

    nhidden+= repmat(paras->hbias, batch_size);
    // activation, sigmloid, backup activation in nhidden

    //std::cout<<"hidden after add base"<<std::endl;
    //copytensor.Resize(paras->Wi2h.shape_);
    //copytensor.Resize(nhidden.shape_);
    //Copy(copytensor, nhidden, nhidden.stream_);
    //display2Tensor(copytensor);

    nhidden = F<cube>(nhidden);

    if(bDropOut){
        TensorContainer<xpu,2, real_t> mask;
        mask.set_stream(paras->stream);
        mask.Resize(nhidden.shape_);
        //paras->rnd.SampleUniform(&mask, 0.0f, 1.0f);
        // F<threshold>(mask, CConfig::fDropoutProb);
        nhidden = nhidden * mask;
    } //dropout
    //std::cout<<"hidden after sigmoid"<<std::endl;
    //copytensor.Resize(paras->Wi2h.shape_);
    //copytensor.Resize(nhidden.shape_);
    //Copy(copytensor, nhidden, nhidden.stream_);
    //display2Tensor(copytensor);


    Copy(nhiddenbak, nhidden, nhiddenbak.stream_);
    // second layer fullc
    nout = dot(nhiddenbak, paras->Wh2o);
    // softmax calculation
  //  Softmax(nout, nout); // in TNN training, we do not need softmax in each step
    // copy result out
    Copy(oubatch, nout, nout.stream_);

    //std::cout<<"nout"<<std::endl;
    //display2Tensor(oubatch);
    //exit(0);
 }

 void display1Tensor( Tensor<cpu, 1, real_t> & tensor ){
     for(int i = 0; i < tensor.size(0); i++)
         std::cout<<tensor[i]<<" ";
     std::cout<<std::endl;
 }

 void display2Tensor( Tensor<cpu, 2, double> tensor ){
     std::cout<<"size 0 :" << tensor.size(0)<<" size 1: "<<tensor.size(1)<<std::endl;
     for(int i = 0; i < tensor.size(0); i++){
        for(int j = 0; j < tensor.size(1); j++)
            std::cout<<tensor[i][j]<<" ";
        std::cout<<std::endl;
     }
 }

 void display2TensorGPU( Tensor<gpu, 2, real_t> & tensor ){
     for(int i = 0; i < tensor.size(0); i++)
        for(int j = 0; j < tensor.size(1); j++)
            std::cout<<tensor[i][j]<<" ";
     std::cout<<std::endl;
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
 }

 /*
  * synchronize the gradients for diffetent threads
  */
 void SubsideGrads(UpdateGrads<xpu>& cumulatedGrads){
    cumulatedGrads.cg_hbias = cumulatedGrads.cg_hbias + g_hbias;
    cumulatedGrads.cg_Wi2h = cumulatedGrads.cg_Wi2h + g_Wi2h;
    cumulatedGrads.cg_Wh2o = cumulatedGrads.cg_Wh2o + g_Wh2o;    
 }

 static void UpdateCumulateGrads(UpdateGrads<xpu>& cgrads, NNetPara<xpu>* paras){
    // run SGD
    
    // l2 regularization
    cgrads.cg_Wi2h += CConfig::fRegularizationRate * paras->Wi2h;
    cgrads.cg_Wh2o += CConfig::fRegularizationRate * paras->Wh2o;
    cgrads.cg_hbias += CConfig::fRegularizationRate * paras->hbias;

    // update weight with adagrad
    paras->eg2Wi2h += F<square>(cgrads.cg_Wi2h);
    paras->eg2Wh2o += F<square>(cgrads.cg_Wh2o);
    paras->eg2Hbias+= F<square>(cgrads.cg_hbias);
    paras->Wi2h -= CConfig::fBPRate * ( cgrads.cg_Wi2h / F<mySqrt>( paras->eg2Wi2h + CConfig::fAdaEps ) );
    paras->Wh2o -= CConfig::fBPRate * ( cgrads.cg_Wh2o / F<mySqrt>( paras->eg2Wh2o + CConfig::fAdaEps ) );
    paras->hbias -= CConfig::fBPRate * ( cgrads.cg_hbias / F<mySqrt>( paras->eg2Hbias + CConfig::fAdaEps ) );

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

#endif
