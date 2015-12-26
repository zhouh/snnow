/*************************************************************************
  > File Name: src/chunker/Model.h
  > Author: Hao Zhou
  > Mail: haozhou0806@gmail.com 
  > Created Time: 26/12/15 15:03:18
 ************************************************************************/

#include "mshadow/tensor.h"
#include "NNet.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

/*
 * parameters of a neural net
 */
template<typename xpu>
class Model{
    Stream<xpu> *stream = NewStream<xpu>();

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
    std::vector<TensorContainer<xpu, 2, real_t>> featEmbs;


    Model(int batch_size, int num_in, int num_hidden, int num_out, 
            std::vector<FeatureType>& featTypes, bool bRndSample) : rnd(0) {

        /*
         * set streams for data
         */
        Wi2h.set_stream(stream);
        Wh2o.set_stream(stream);
        hbias.set_stream(stream);

        /*
         * initialize the matries in neural net
         */
        Wi2h.Resize(Shape2(num_in, num_hidden), static_cast<real_t>(0.0));  
        Wh2o.Resize(Shape2(num_hidden, num_out), static_cast<real_t>(0.0)); 
        hbias.Resize(Shape1(num_hidden), static_cast<real_t>(0.0)); 


        // if the model is the gradient square, we just need to set them 0
        if(bRndSample){ 
            rnd.SampleUniform(&Wi2h, -1.0 * CConfig::fInitRange, CConfig::fInitRange);
            rnd.SampleUniform(&Wh2o, -0.01, 0.01f);
            rnd.SampleUniform(&hbias, -0.01, 0.01f);
        }

        /*
         * initialize the feature embeddings
         * the feature embedding need to be all zeros here
         */
        featEmbs.resize(featTypes.size());
        for(int i = 0; i < featTypes.size(); i++){

            featEmbs[i].set_stream(stream);
            featEmbs[i].Resize(Shape2(featTypes[i].totalFeatNum, featTypes[i].dim), static_cast<real_t>(0.0));
        }

    }

    /**
     * set all the paras in the mode as zero
     */
    void setZero(){
        Wi2h = 0.0;
        Wh2o = 0.0;
        hbias = 0.0;

        for(int i = 0; i < featEmbs.size(); i++)
            featEmbs[i] = 0.0;

    }

    /**
     * update the given gradients and adagrad square sums for this model
     * with adaGrad Updating and l2-regularization
     */
    void update(Model<xpu>* gradients, Model<xpu>* adaGradSquares){

        // l2 regularization
        gradients->Wi2h += CConfig::fRegularizationRate * Wi2h;
        gradients->Wh2o += CConfig::fRegularizationRate * Wh2o;
        gradients->hbias += CConfig::fRegularizationRate * hbias;
        for(int i = 0; i < featEmbs.size(); i++)
            gradients->featEmbs[i] += CConfig::fRegularizationRate * featEmbs[i];

        // update adagrad gradient square sums
        adaGradSquares->Wi2h += F<square>(gradients->Wi2h);
        adaGradSquares->Wh2o += F<square>(gradients->Wh2o);
        adaGradSquares->hbias+= F<square>(gradients->hbias);
        for(int i = 0; i < gradients->featEmbs.size(); i++)
            adaGradSquares->FeatEmbs += F<square>(gradients->featEmbs);

        // update this weight
        Wi2h -= CConfig::fBPRate * ( gradients->Wi2h / F<mySqrt>( adaGradSquares->Wi2h + CConfig::fAdaEps ) );
        Wh2o -= CConfig::fBPRate * ( gradients->Wh2o / F<mySqrt>( adaGradSquares->Wh2o + CConfig::fAdaEps ) );
        hbias -= CConfig::fBPRate * ( gradients->hbias / F<mySqrt>( adaGradSquares->hbias + CConfig::fAdaEps ) );
        for(int i = 0; i < gradients->featEmbs.size(); i++)
            featEmbs[i] -= CConfig::fBPRate * ( gradients->featEmbs[i] / F<mySqrt>( adaGradSquares->FeatEmbs[i] + CConfig::fAdaEps ) );

        gradients->setZeros(); // set zero for that the cumulated gradients will be reused in the next update
    }

    /**
     * merge two models, used for merge model gradients
     */
    void mergeMode(Model * model){
        Wi2h += model->Wi2h;
        Wh2o += model->Wh2o;
        hbias += model->hbias;
        for(int i = 0; i < featEmbs.size(); i++)
            featEmbs[i] +=  model->featEmbs[i];

    }

    /**
     * #TODO fill the function
     * convert the input gradients obtained from the neural network
     * to the feature embedding gradients according to the corresponding feature vector
     */
    void inputGradient2FeatEmbGradient(FeatureVector& fv, TensorContainer<1, real_t> netInputGradient){
        
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
