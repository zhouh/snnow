/*
 * Depparser.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#include <omp.h>
#include <random>
#include <algorithm>

#include "Depparser.h"
#include "State.h"
#include "Config.h"
#include "mshadow/tensor.h"
#include "NNet.h"
#include "FeatureEmbedding.h"
#include "FeatureExtractor.h"

using namespace mshadow;
using namespace mshadow::expr;

Depparser::Depparser(bool bTrain) {
	/*beamSize = CConfig::nBeamSize;*/
	/*m_bTrain = bTrain;*/
}

Depparser::~Depparser() {
}

void Depparser::train(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees,
        std::vector<DepParseInput> devInputs, std::vector<DepTree> devTrees) {

    std::cout<<"Training begin!"<<std::endl;
    std::cout<<"Training Instance Num: "<<inputs.size()<<std::endl;

    /*Prepare the feature extractor*/
    featExtractor.getDictionaries(goldTrees);
    featExtractor.generateTrainingExamples(inputs, goldTrees, gExamples);

    // prepare for the neural networks, every parsing step maintains a specific net
    // because each parsing step has different updating gradients.
    const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
    const int num_hidden = CConfig::nHiddenSize;
    const int num_out = kActNum;
    const int beamSize = CConfig::nBeamSize;
    omp_set_num_threads(CConfig::nThread);  //set the threads for mini-batch learning
    srand(0);
    NNet<gpu>::init(beamSize, num_in, num_hidden, num_out);	//init the static member in the neural net
    FeatureEmbedding<cpu> fEmb(CConfig::nFeatureNum, CConfig::nEmbeddingDim, beamSize);

    // for every iteration
    for(int iter = 0; iter < CConfig::nRound; iter++){
        /*
         * randomly sample the training instances in the container,
         * and assign them for each thread
         */
        std::vector<std::vector<GlobalExample*>> multiThread_miniBtach_data;

        //get mini-batch data for each threads
        std::random_shuffle ( gExamples.begin(), gExamples.end() );
        int threadExampleNum = CConfig::nBatchSize / CConfig::nThread;
        auto sp = gExamples.begin();
        auto ep = sp + threadExampleNum;
        for(int i = 0; i < CConfig::nThread; i++){
            std::vector<GlobalExample*> threadExamples;
            for(auto p = sp; p != ep; p++){
                threadExamples.push_back( &( *p ) );
            }
            sp = ep;
            ep += threadExampleNum;
            multiThread_miniBtach_data.push_back(threadExamples);
        }

        //set up mshadow tensor
        InitTensorEngine<gpu>();

        // begin to multi-thread training
#pragma omp parallel
        {
            auto currentThreadData = multiThread_miniBtach_data[omp_get_thread_num()];

            // temp input layer
            TensorContainer<cpu, 2> input;
            input.Resize( Shape2( beamSize, num_in ) );
            // temp output layer
            TensorContainer<cpu, 2> pred;
            pred.Resize( Shape2( beamSize, num_out ) );

            //for every instance
            for(unsigned inst = 0; inst < currentThreadData.size(); inst++){
                //get current training instance
                GlobalExample * example =  currentThreadData[inst];
                const int sentLen = example->wordIdx.size();
                const int maxRound = sentLen * 2 + 1;
                const int max_lattice_size =  (beamSize + 1) * maxRound;
//				int num_results = 0;
                int round = 0;
                int currentBeamSize = 1; // initially, the beam only have one empty state
                int correctStateIdx;
                bool bBeamContainGold = true;
                double maxScore = 0;
                Beam beam(beamSize);

                std::vector<NNet<gpu>*> nets;

                if(inst % 1000 == 0)
                    std::cout<<"Processing sentence "<<inst<<std::endl;
                // beam search decoding
                State * lattice = new State[max_lattice_size];
                State * lattice_index[maxRound];
                State * correctState = lattice;
                for (int i = 0; i < max_lattice_size; ++i) {
                    lattice[i].len_ = sentLen;
                }

                lattice[0].clear();
                lattice[0].setBeamIdx(0);
                correctState = lattice;
                lattice_index[0] = lattice;
                lattice_index[1] = lattice_index[0] + 1;

                // for every round in training
//				int beamIdx = 0;
                for(round = 1; round < maxRound; round++){

                    NNet<gpu> *net = new NNet<gpu>(beamSize, num_in, num_hidden, num_out);
                    nets.push_back(net);
                    // new round, set beam gold false
                    bBeamContainGold = false;
                    // extract feature vectors in batch
                    std::vector<std::vector<int> > featureVectors(currentBeamSize);
                    getInputBatch(lattice_index[round - 1], example->wordIdx,
                                  example->tagIdx, featureVectors);
                    fEmb.returnInput(featureVectors, input);
                    net->Forward(input, pred);

                    // for every state in the last beam, expand and insert into next beam
                    int stateIdx = 0;
                    for (State * currentState = lattice_index[round - 1];
                            currentState != lattice_index[round]; ++currentState, ++stateIdx) {
                        std::vector<int> validActs;
                        currentState->getValidActs(validActs);

                        //for every valid action
                        for(unsigned actID = 0; actID < validActs.size(); ++actID){
                            //skip invalid action
                            if(validActs[actID] == -1)
                                continue;
                            //construct scored transition, and insert into beam
                            CScoredTransition trans;
                            trans(currentState, actID, currentState->score + pred[stateIdx][actID]);
                            beam.insert(trans);
                            currentBeamSize = ( currentBeamSize + 1 ) >= beamSize ? beamSize : ( currentBeamSize + 1 );
                        } // valid action #for end

                        //lazy expand the states in the beam
                        for (int i = 0; i < beam.currentBeamSize; ++i) {
                            const CScoredTransition& transition = beam.beam[i];
                            State* target = lattice_index[round] + i;
                            target->copy( *(transition.source) );
                            // generate candidate state according to the states in beam
                            target->Move(transition.action);
                            target->setBeamIdx(i);
                            target->score = transition.score;
                            target->previous_ = transition.source;
                            target->bGold = target->previous_->bGold & target->last_action == example->goldActs[round - 1]; // beam states contain gold state ?  bBeamContainGold |= target->bGold;

                            if(target->bGold == true){
                                correctState = target;
                                correctStateIdx = i;
                            }
                            if( i == 0 || target->score > maxScore )
                                maxScore = target->score;
                        }
                    } // beam #for end

                    if( bEarlyUpdate & !bBeamContainGold & m_bTrain)
                        break;

                    // prepare lattice for next parsing round
                    lattice_index[round + 1] = lattice_index[round] + currentBeamSize;
                } //round #for end

                // update parameter
                if (m_bTrain) {

                    std::vector<State*> trainingStates;
                    for(int bi = 0; bi < currentBeamSize; ++bi){
                        trainingStates.push_back( beam.beam[bi].source );
                    }
                    /* With early update, now the gold state fall out beam,*/
                    /* we need to expand the gold state one more step.*/
                    if( bEarlyUpdate & !bBeamContainGold ){
                        State* next_correct_state = lattice_index[round] + currentBeamSize;
                        next_correct_state->copy(*correctState);
                        next_correct_state->Move(example->goldActs[round - 1]);
                        next_correct_state->previous_ = correctState;
                        correctState = next_correct_state;
                        //endLatice = correctState;
                        correctStateIdx = currentBeamSize;
                        trainingStates.emplace_back(correctState);
                    }
                    /*computes the gradients of beam contrastive learning*/
                    int trainingDataSize = trainingStates.size();
                    std::vector<float> updateParas(trainingDataSize, 0); // updating parameter vector
                    // softmax
                    double sum =0;
                    for (int b_j = 0; b_j < trainingDataSize; b_j++) {
                        updateParas[b_j] = exp( trainingStates[b_j]->score - maxScore );
                        sum += updateParas[b_j];
                    }
                    for (int b_j = 0; b_j < trainingDataSize; b_j++) {
                        updateParas[b_j] = updateParas[b_j] / sum;
                        sum += updateParas[b_j];
                    }
                    updateParas[correctStateIdx] -= 1.0;

                    /*  Back propagation updating,*/
                    /*  from last parsing state to the former states*/
                    for(int backRound = round; backRound > 0; --backRound){
                        TensorContainer<cpu, 2> grads;
                        input.Resize(Shape2(beamSize, num_out));
                        int i = 0;
                        for(auto iter = trainingStates.begin(); iter != trainingStates.end(); iter++, i++){
                            grads[ ( *iter )->previous_->beamIdx ][ ( *iter )->last_action ] = updateParas[i];
                            *iter = ( *iter )->previous_;
                        }
                        nets[backRound - 1]->Backprop(grads);
                    }

                    NNet<gpu>::Update();

                } // updating end
                else{ // in testing
                    // get best expanded state
                    State * bestState = lattice_index[round];
                    for (State * p = lattice_index[round]; p != lattice_index[round + 1]; ++p) {
                        if (bestState->score < p->score) {
                            bestState = p;
                        }
                    }
                } // testing end

            } // instance #for end

        } // end multi-processor
        ShutdownTensorEngine<gpu>();

    } // iteration #for end

}

void Depparser::parse(std::vector<DepParseInput> inputs) {
}


   //get the feature vector in all the beam states,
   //and return the input layer of neural network in a batch.
void Depparser::getInputBatch(State* state, std::vector<int>& wordIndexCache,
        std::vector<int>& tagIndexCache,
        std::vector<std::vector<int> >& featvecs) {

    for(unsigned i = 0; i < featvecs.size(); i++){
        std::vector<int> featvec(CConfig::nFeatureNum);
        featExtractor.featureExtract( state + i, wordIndexCache, tagIndexCache, featvec);
        featvecs.push_back(featvec);
    }
}


