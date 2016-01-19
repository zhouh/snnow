/*************************************************************************
  > File Name: src/chunker/Model.h
  > Author: Hao Zhou
  > Mail: haozhou0806@gmail.com 
  > Created Time: 26/12/15 15:03:18
 ************************************************************************/

#ifndef _CHUNKER_MODEL_H_
#define _CHUNKER_MODEL_H_

#include <memory>

#include "mshadow/tensor.h"

#include "chunker.h"

#include "Config.h"
#include "FeatureEmbeddingManager.h"
#include "FeatureEmbedding.h"
#include "FeatureType.h"

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

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
class Model{
public:
    Stream<xpu> *stream;
    /*
     * parameters for a neural network
     */
    TensorContainer<xpu, 2, real_t> Wi2h, Wh2o;
    TensorContainer<xpu, 1, real_t> hbias;

    /*
     * random seed generator
     */
    Random<xpu, real_t> rnd; 

    /*
     * parameters of feature embeddings
     */
    std::shared_ptr<FeatureEmbeddingManager> featEmbManagerPtr;
    std::vector< std::shared_ptr<FeatureEmbedding> > featEmbs;
    std::vector<FeatureType> featTypes;

    Model(int num_in, int num_hidden, int num_out, 
          std::shared_ptr<FeatureEmbeddingManager> fEmbManagerPtr, Stream<xpu> *stream, bool bRndInitialize = false) : rnd(0), featEmbManagerPtr(fEmbManagerPtr) {
        featTypes = featEmbManagerPtr->getFeatureTypes();
        /*
         * set streams for data
         */
        // this->stream = NewStream<xpu>();
        this->stream = stream;

        Wi2h.set_stream(this->stream);
        Wh2o.set_stream(this->stream);
        hbias.set_stream(this->stream);

        /*
         * initialize the matries in neural net
         */
        Wi2h.Resize(Shape2(num_in, num_hidden), static_cast<real_t>(0.0));  
        Wh2o.Resize(Shape2(num_hidden, num_out), static_cast<real_t>(0.0)); 
        hbias.Resize(Shape1(num_hidden), static_cast<real_t>(0.0)); 


        // if the model is the gradient square, we just need to set them 0
        if(bRndInitialize){ 
            rnd.SampleUniform(&Wi2h,  -1.0 * CConfig::fInitRange, CConfig::fInitRange);
            rnd.SampleUniform(&Wh2o,  -1.0 * CConfig::fInitRange, CConfig::fInitRange);
            rnd.SampleUniform(&hbias, -1.0 * CConfig::fInitRange, CConfig::fInitRange);
        }

        /*
         * initialize the feature embeddings
         * the feature embedding need to be all zeros here
         */
        if (bRndInitialize) {
            featEmbs = featEmbManagerPtr->getInitialzedEmebddings(CConfig::fInitRange);
            if (CConfig::loadModel) {
                std::fstream is(CConfig::strNetModelPath);
                loadModel(is);
            }
        } else {
            featEmbs = featEmbManagerPtr->getAllZeroEmebddings();
        }
    }

    // Model(const Model<xpu> &model) : rnd(0), featTypes(model.featTypes), featEmbs(model.featEmbs){
    //     this->stream = NewStream<xpu>();

    //     Wi2h.set_stream(this->stream);
    //     Wh2o.set_stream(this->stream);
    //     hbias.set_stream(this->stream);

    //     Wi2h = model.Wi2h;
    //     Wh2o = model.Wh2o;
    //     hbias = model.hbias;
    // }

    // Model<xpu>& operator=(const Model<xpu> &model) {
    //     if (this == &model) {
    //         return *this;
    //     }

    //     featTypes = model.featTypes;
    //     featEmbs = model.featEmbs;

    //     this->stream = NewStream<xpu>();

    //     Wi2h.set_stream(this->stream);
    //     Wh2o.set_stream(this->stream);
    //     hbias.set_stream(this->stream);

    //     Wi2h = model.Wi2h;
    //     Wh2o = model.Wh2o;
    //     hbias = model.hbias;

    //     return *this;
    // }

    ~Model() {
        // DeleteStream(stream);
    }

    /**
     * set all the paras in the mode as zero
     */
    void setZero(){
        Wi2h = 0.0;
        Wh2o = 0.0;
        hbias = 0.0;

        for(int i = 0; i < featEmbs.size(); i++)
            featEmbs[i]->setZero();
    }

    real_t norm2() {
        real_t res = 0.0;

        static TensorContainer<cpu, 2, real_t> Wi2h_cpu(Wi2h.shape_);
        static TensorContainer<cpu, 2, real_t> Wh2o_cpu(Wh2o.shape_);
        static TensorContainer<cpu, 1, real_t> hbias_cpu(hbias.shape_);


        Copy(Wi2h_cpu, Wi2h, Wi2h.stream_);
        for (int i = 0; i < Wi2h.shape_[0]; i++) {
            for (int j = 0; j < Wi2h.shape_[1]; j++) {
                res += Wi2h_cpu[i][j] * Wi2h_cpu[i][j];
            }
        }

        Copy(Wh2o_cpu, Wh2o, Wh2o.stream_);
        for (int i = 0; i < Wh2o.shape_[0]; i++) {
            for (int j = 0; j < Wh2o.shape_[1]; j++) {
                res += Wh2o_cpu[i][j] * Wh2o_cpu[i][j];
            }
        }

        Copy(hbias_cpu, hbias, hbias.stream_);
        for (int i = 0; i < hbias.shape_[0]; i++) {
            res += hbias_cpu[i] * hbias_cpu[i];
        }

        for (int fi = 0; fi < featEmbs.size(); fi++) {
            auto &featEmb = featEmbs[fi];

            for (int i = 0; i < featEmb->dictSize; i++) {
                for (int j = 0; j < featEmb->embeddingSize; j++) {
                    res += featEmb->data[i][j] * featEmb->data[i][j];
                }
            }
        }

        return res;
    }

    real_t embeddings_norm2() {
        real_t res = 0.0;

        for (int fi = 0; fi < featEmbs.size(); fi++) {
            auto &featEmb = featEmbs[fi];

            for (int i = 0; i < featEmb->dictSize; i++) {
                for (int j = 0; j < featEmb->embeddingSize; j++) {
                    res += featEmb->data[i][j] * featEmb->data[i][j];
                }
            }
        }

        return res;
    }
    /**
     * update the given gradients and adagrad square sums for this model
     * with adaGrad Updating and l2-regularization
     */
    void update(Model<xpu>* gradients, Model<xpu>* adaGradSquares){
        // l2 regularization
        gradients->Wi2h  += CConfig::fRegularizationRate * Wi2h;
        gradients->Wh2o  += CConfig::fRegularizationRate * Wh2o;
        gradients->hbias += CConfig::fRegularizationRate * hbias;
        if (CConfig::bFineTune) {
            for(int i = 0; i < featEmbs.size(); i++)
                gradients->featEmbs[i]->data += CConfig::fRegularizationRate * featEmbs[i]->data;
        }

        // update adagrad gradient square sums
        adaGradSquares->Wi2h  += F<square>(gradients->Wi2h);
        adaGradSquares->Wh2o  += F<square>(gradients->Wh2o);
        adaGradSquares->hbias += F<square>(gradients->hbias);
        if (CConfig::bFineTune) {
            for(int i = 0; i < gradients->featEmbs.size(); i++)
                adaGradSquares->featEmbs[i]->data += F<square>(gradients->featEmbs[i]->data);
        }

        // update this weight
        Wi2h  -= CConfig::fBPRate * ( gradients->Wi2h  / F<mySqrt>( adaGradSquares->Wi2h + CConfig::fAdaEps  ) );
        Wh2o  -= CConfig::fBPRate * ( gradients->Wh2o  / F<mySqrt>( adaGradSquares->Wh2o + CConfig::fAdaEps  ) );
        hbias -= CConfig::fBPRate * ( gradients->hbias / F<mySqrt>( adaGradSquares->hbias + CConfig::fAdaEps ) );
        if (CConfig::bFineTune) {
            for(int i = 0; i < gradients->featEmbs.size(); i++)
                featEmbs[i]->data -= CConfig::fBPRate * ( gradients->featEmbs[i]->data / F<mySqrt>( adaGradSquares->featEmbs[i]->data + CConfig::fAdaEps ) );
        }

        gradients->setZero(); // set zero for that the cumulated gradients will be reused in the next update
    }

    /**
     * merge two models, used for merge model gradients
     */
    void mergeModel(Model * model){
        Wi2h  += model->Wi2h;
        Wh2o  += model->Wh2o;
        hbias += model->hbias;
        if (CConfig::bFineTune) {
            for(int i = 0; i < featEmbs.size(); i++)
                featEmbs[i]->data +=  model->featEmbs[i]->data;
        }
    }

    /**
     * The save and read model module need to be rewrite
     */
    void saveModel( std::ostream & os ){
        TensorContainer<cpu, 2, real_t> sWi2h(Wi2h.shape_);
        TensorContainer<cpu, 2, real_t> sWh2o(Wh2o.shape_);
        TensorContainer<cpu, 1, real_t> shbias(hbias.shape_);

        Copy(sWi2h, Wi2h, Wi2h.stream_);
        Copy(sWh2o, Wh2o, Wh2o.stream_);
        Copy(shbias, hbias, hbias.stream_);

        /*
         * write the Wi2h
         */
        os << sWi2h.size(0) << "\t" << sWi2h.size( 1 ) << std::endl;
        for( index_t i = 0; i < sWi2h.size( 0 ); i++ )
            for( index_t j = 0; j < sWi2h.size( 1 ); j++ ){
                os << sWi2h[ i ][ j ];
                if( j == ( sWi2h.size( 1 ) - 1 ) )
                    os << std::endl;
                else
                    os << " ";
            }

        /*
         * write the Wh2o
         */
        os << sWh2o.size(0) << "\t" << sWh2o.size( 1 ) << std::endl;
        for( index_t i = 0; i < sWh2o.size( 0 ); i++ )
            for( index_t j = 0; j < sWh2o.size( 1 ); j++ ){
                os << sWh2o[ i ][ j ];
                if( j == ( sWh2o.size( 1 ) - 1 ) )
                    os << std::endl;
                else
                    os << " ";
            }

        /*
         * write the hbias
         */
        os << shbias.size(0) << std::endl;
        for( index_t i = 0; i < shbias.size( 0 ); i++ ){
            os << shbias[ i ];
            if(  i == ( shbias.size( 0 ) - 1 ) )
                os << std::endl;
            else
                os << " ";
        }

        featEmbManagerPtr->saveFeatureEmbeddings(featEmbs);
    }

    void loadModel( std::istream & is ){
        TensorContainer<cpu, 2, real_t> sWi2h(Wi2h.shape_);
        TensorContainer<cpu, 2, real_t> sWh2o(Wh2o.shape_);
        TensorContainer<cpu, 1, real_t> shbias(hbias.shape_);

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
                iss_j >> sWi2h[ i ][ j ];
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
                iss_wh2o_j >> sWh2o[ i ][ j ];
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
            iss_hbias_j >> shbias[ i ];
        }

        Copy(Wi2h, sWi2h, Wi2h.stream_);
        Copy(Wh2o, sWh2o, Wh2o.stream_);
        Copy(hbias, shbias, hbias.stream_);

        featEmbs = featEmbManagerPtr->loadFeatureEmbeddings();
    }
private:
    Model(const Model<xpu> &model) = delete;
    Model<xpu>& operator= (const Model<xpu> &model) = delete;
};
// template<typename xpu>
// Random<xpu, real_t>  Model<xpu>::rnd(0);

#endif
