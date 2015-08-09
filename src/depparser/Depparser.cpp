/*
 * Depparser.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#include <omp.h>
#include <unordered_set>
#include <random>
#include <algorithm>

#include "math.h"
#include "Depparser.h"
#include "State.h"
#include "Config.h"
#include "mshadow/tensor.h"
#include "NNet.h"

using namespace mshadow;
using namespace mshadow::expr;

Depparser::Depparser(bool bTrain) {
	beamSize = CConfig::nBeamSize;
	m_bTrain = bTrain;
}

Depparser::~Depparser() {
}

void Depparser::train(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees,
		std::vector<DepParseInput> devInputs, std::vector<DepTree> devTrees) {

	std::cout<<"Training begin!"<<std::endl;
	std::cout<<"Training Instance Num: "<<inputs.size()<<std::endl;

	//get dictionary
	getDictionaries(goldTrees);
	generateTrainingExamples(inputs, goldTrees);

	// settings for neural network
	const int num_in = CConfig::nEmbeddingDim * CConfig::nFeatureNum;
	const int num_hidden = CConfig::nHiddenSize;
	const int num_out = kActNum;
	const int beamSize = CConfig::nBeamSize;

	omp_set_num_threads(CConfig::nThread);  //set the threads for mini-batch learning
	srand(0);
	NNet::init(beamSize, num_in, num_hidden, num_out);	//init the static member in the neural net
	FeatureEmbedding fEmb(CConfig::nFeatureNum, CConfig::nEmbeddingDim, beamSize);

	// for every iteration
	for(int iter = 0; iter < CConfig::nRound; iter++){
		/*
		 *  randomly sample the training instances in the container,
		 *  and assign them for each thread
		 */
		std::vector<std::vector<GlobalExample*>> multiThread_miniBtach_data;

		//get mini-batch data for each threads
		std::random_shuffle ( gExamples.begin(), gExamples.end() );
		int threadExampleNum = CConfig::nBatchSize / CConfig::nThread;
		auto sp = gExamples.begin();
		auto ep = sp + threadExampleNum;
		for(int i = 0; i < CConfig::nThread; i++){
			std::vector<GlobalExample*> threadExamples(sp, ep);
			sp = ep;
			ep += threadExampleNum;
		}

		//set up mshadow tensor
		InitTensorEngine();

		// begin to multi-thread training
#pragma omp parallel
		{
			auto currentThreadData = multiThread_miniBtach_data[omp_get_thread_num()];

			// temp input layer
			TensorContainer<cpu, 2> input;
			input.Resize( Shape2( beamSize, num_out ) );
			// temp output layer
			TensorContainer<cpu, 2> pred;
			pred.Resize( Shape2( beamSize, num_out ) );

			//for every instance
			for(int inst = 0; inst < currentThreadData.size(); inst++){
				//get current training instance
				GlobalExample * example =  currentThreadData[inst];
				const int sentLen = example->wordIdx.size();
				const int maxRound = sentLen * 2 + 1;
				const int max_lattice_size =  (beamSize + 1) * maxRound;
				int num_results = 0;
				int round = 0;
			    int currentBeamSize = 1; // initially, the beam only have one empty state
			    int correctStateIdx;
				bool bBeamContainGold = true;
				double maxScore = 0;
				CScoredTransition* beamTransits = new CScoredTransition[beamSize];

				std::vector<NNet*> nets;

				if(inst % 1000 == 0)
					std::cout<<"Processing sentence "<<inst<<std::endl;
				// beam search decoding
				State * lattice = new State[max_lattice_size];
				State * lattice_index = new State[maxRound];
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
				int beamIdx = 0;
			    for(round = 1; round < maxRound; round++){

			    	NNet *net = new NNet<cpu>(beamSize, num_in, num_hidden, num_out);
			    	nets.emplace_back(net);
			    	// new round, set beam gold false
			    	bBeamContainGold = false;
			    	// extract feature vectors in batch
					std::vector<std::vector<int> > featureVectors(
							currentBeamSize);
					getInputBatch(lattice_index[round - 1], example->wordIdx,
							example->tagIdx, featureVectors);
					fEmb.returnInput(featureVectors, input);
					net->Forward(input, pred);

			    	// for every state in the last beam, expand and insert into next beam
					int stateIdx = 0;
					currentBeamSize = 0; //clear all the beam state
					for (State * currentState = lattice_index[round - 1];
							currentState != lattice_index[round]; ++currentState, ++stateIdx) {
						std::vector<int> validActs;
						currentState->getValidActs(validActs);

						//for every valid action
						for(int actID = 0; actID < validActs.size(); ++actID){
							//skip invalid action
							if(validActs[actID] == -1)
								continue;
							//construct scored transition, and insert into beam
							CScoredTransition trans;
							trans(currentState, actID, currentState->score + pred[stateIdx][actID]);
							insertBeam(trans, beamTransits, currentBeamSize);
							currentBeamSize = ( currentBeamSize + 1 ) >= beamSize ? beamSize : ( currentBeamSize + 1 );
						} // valid action #for end

						//lazy expand the states in the beam
						for (unsigned i = 0; i < currentBeamSize; ++i) {
							const CScoredTransition& transition = beamTransits[i];
							State* target = lattice_index[round] + i;
							target->copy( *(transition.source) );
							// generate candidate state according to the states in beam
							target->Move(transition.action);
							target->setBeamIdx(i);
							target->score = transition.score;
							target->previous_ = transition.source;
							target->bGold = target->previous_->bGold
											& target->last_action == example->goldActs[round - 1];
							// beam states contain gold state ?
							bBeamContainGold |= target->bGold;

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

				/*
				 * update parameter
				 */
				if (m_bTrain) {

					std::vector<State> trainingStates(beamTransits, beamTransits + currentBeamSize);
					/*
					 *  With early update, now the gold state fall out beam,
					 *  we need to expand the gold state one more step.
					 */
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
					/*
					 *   computes the gradients of beam contrastive learning
					 */
					int trainingDataSize = trainingStates.size();
					std::vector<float> updateParas(trainingDataSize, 0); // updating parameter vector
					// softmax
					double sum =0;
					for (int b_j = 0; b_j < trainingDataSize; b_j++) {
						updateParas[b_j] = exp( trainingStates[b_j].score - maxScore );
						sum += updateParas[b_j];
					}
					for (int b_j = 0; b_j < trainingDataSize; b_j++) {
						updateParas[b_j] = updateParas[b_j] / sum;
						sum += updateParas[b_j];
					}
					updateParas[correctStateIdx] -= 1.0;

					/*
					 *  Back propagation updating,
					 *  from last parsing state to the former states
					 */
					for(int backRound = round; backRound > 0; --backRound){
						TensorContainer<cpu, 2> grads;
						input.Resize(Shape2(beamSize, num_out));
						int i = 0;
						for(auto iter = trainingStates.begin(); iter != trainingStates.end(); iter++, i++){
							grads[ iter->previous_->beamIdx ][ iter->last_action ] = updateParas[i];
							iter = iter->previous_;
						}

						nets[round - 1]->Backprop(grads);
						nets[round - 1]->Update();
					}

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
		ShutdownTensorEngine();

	} // iteration #for end


}

int

bool
ScoredTransitionMore(const CScoredTransition& x, const CScoredTransition& y) {
  return x.score > y.score;
}

void Depparser::init(std::vector<DepParseInput> inputs,
		std::vector<DepTree> goldTrees) {

	getDictionaries(goldTrees);
	generateTrainingExamples(inputs, goldTrees);

}

void Depparser::parse(std::vector<DepParseInput> inputs) {
}

/**
 *   get the dictionary
 */
void Depparser::getDictionaries(std::vector<DepTree> goldTrees) {

	std::unordered_set labelSet;
	std::unordered_set tagSet;
	std::unordered_set wordSet;

	// insert to the set
	for (int i = 0; i < goldTrees.size(); i++) {
		for (int j = 0; j < goldTrees[i].size; j++) {
			auto treeNode = goldTrees[i].nodes[j];
			labelSet.insert(treeNode.label);
			tagSet.insert(treeNode.tag);
			wordSet.insert(treeNode.word);
		}
	}

	// insert into the map
	int index = 0;
	//label
	for (auto it = labelSet.begin(); it != labelSet.end(); ++it) {
		labelMap[*it] = index++;
		knowLabels.push_back(*it);
	}
	//knowLabels.push_back(null); we do not need null in knowLabel!
	//the size of knowLabel will be used for shift-reduce actions
	labelNullIdx = index;
	labelMap[null] = index++;

	//tag
	tagUnkIdx = index;
	tagMap[unknow] = index++;
	tagRootIdx = index;
	tagMap[root] = index++;
	tagNullIdx = index;
	tagMap[null] = index++;
	knowTags.push_back(unknow);
	knowTags.push_back(root);
	knowTags.push_back(null);
	for (auto it = tagSet.begin(); it != tagSet.end(); ++it) {
		tagMap[*it] = index++;
		knowTags.push_back(*it);
	}
	//word
	wordUnkIdx = index;
	wordMap[unknow] = index++;
	wordRootIdx = index;
	wordMap[root] = index++;
	wordNullIdx = index;
	wordMap[null] = index++;
	knowWords.push_back(unknow);
	knowWords.push_back(root);
	knowWords.push_back(null);
	for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
		wordMap[*it] = index++;
		knowWords.push_back(*it);
	}
}

void Depparser::featureExtract(State* state, std::vector<int>& wordIndexCache,
		std::vector<int>& tagIndexCache, std::vector<int> & features) {
	// positions 0-17 hold fWord, 18-35 hold fPos, 36-47 hold fLabel
	int POS_OFFSET = 18;
	int DEP_OFFSET = 36;
	int STACK_OFFSET = 6;
	int STACK_NUMBER = 6;

	int index = 0;
	int IDIdx = 0;

	int s0 = state->stacktop();
	int s1 = state->stack2top();
	int s2 = state->stack3top();
	int q0 = state->m_nNextWord >= state->len_ ? -1 : state->m_nNextWord;
	int q1 =
			(state->m_nNextWord + 1) >= state->len_ ?
					-1 : (state->m_nNextWord + 1);
	int q2 =
			(state->m_nNextWord + 2) >= state->len_ ?
					-1 : (state->m_nNextWord + 2);

	// words
	features[IDIdx++] = getWordIndex(s0, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1, wordIndexCache);
	features[IDIdx++] = getWordIndex(s2, wordIndexCache);
	features[IDIdx++] = getWordIndex(q0, wordIndexCache);
	features[IDIdx++] = getWordIndex(q1, wordIndexCache);
	features[IDIdx++] = getWordIndex(q2, wordIndexCache);

	features[IDIdx++] = getTagIndex(s0, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1, tagIndexCache);
	features[IDIdx++] = getTagIndex(s2, tagIndexCache);
	features[IDIdx++] = getTagIndex(q0, tagIndexCache);
	features[IDIdx++] = getTagIndex(q1, tagIndexCache);
	features[IDIdx++] = getTagIndex(q2, tagIndexCache);

	// s0l s0r s0l2 s0r2 s0ll s0rr
	int s0l, s0r, s0l2, s0r2, s0ll, s0rr;
	s0l = state->leftdep(s0);
	s0r = state->rightdep(s0);
	s0l2 = state->left2dep(s0);
	s0r2 = state->left2dep(s0);
	s0ll = s0l == -1 ? -1 : state->leftdep(s0l);
	s0rr = s0r == -1 ? -1 : state->rightdep(s0r);

	features[IDIdx++] = getWordIndex(s0l, wordIndexCache);
	features[IDIdx++] = getWordIndex(s0r, wordIndexCache);
	features[IDIdx++] = getWordIndex(s0l2, wordIndexCache);
	features[IDIdx++] = getWordIndex(s0r2, wordIndexCache);
	features[IDIdx++] = getWordIndex(s0ll, wordIndexCache);
	features[IDIdx++] = getWordIndex(s0rr, wordIndexCache);

	features[IDIdx++] = getTagIndex(s0l, tagIndexCache);
	features[IDIdx++] = getTagIndex(s0r, tagIndexCache);
	features[IDIdx++] = getTagIndex(s0l2, tagIndexCache);
	features[IDIdx++] = getTagIndex(s0r2, tagIndexCache);
	features[IDIdx++] = getTagIndex(s0ll, tagIndexCache);
	features[IDIdx++] = getTagIndex(s0rr, tagIndexCache);

	features[IDIdx++] = getLabelIndex(s0l, state);
	features[IDIdx++] = getLabelIndex(s0r, state);
	features[IDIdx++] = getLabelIndex(s0l2, state);
	features[IDIdx++] = getLabelIndex(s0r2, state);
	features[IDIdx++] = getLabelIndex(s0ll, state);
	features[IDIdx++] = getLabelIndex(s0rr, state);

	// s1l s1r s1l2 s1r2 s1ll s1rr
	int s1l, s1r, s1l2, s1r2, s1ll, s1rr;
	s1l = state->leftdep(s1);
	s1r = state->rightdep(s1);
	s1l2 = state->left2dep(s1);
	s1r2 = state->left2dep(s1);
	s1ll = s1l == -1 ? -1 : state->leftdep(s1l);
	s1rr = s1r == -1 ? -1 : state->rightdep(s1r);

	features[IDIdx++] = getWordIndex(s1l, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1r, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1l2, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1r2, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1ll, wordIndexCache);
	features[IDIdx++] = getWordIndex(s1rr, wordIndexCache);

	features[IDIdx++] = getTagIndex(s1l, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1r, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1l2, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1r2, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1ll, tagIndexCache);
	features[IDIdx++] = getTagIndex(s1rr, tagIndexCache);

	features[IDIdx++] = getLabelIndex(s1l, state);
	features[IDIdx++] = getLabelIndex(s1r, state);
	features[IDIdx++] = getLabelIndex(s1l2, state);
	features[IDIdx++] = getLabelIndex(s1r2, state);
	features[IDIdx++] = getLabelIndex(s1ll, state);
	features[IDIdx++] = getLabelIndex(s1rr, state);

}

/**
 *   generate training examples for global learning
 */
void Depparser::generateTrainingExamples(std::vector<DepParseInput> inputs,
		std::vector<DepTree> goldTrees) {

	/*
	 * get training examples
	 */
	gExamples.resize(inputs.size());  //resize vector global examples

	//for every sentence
	for (int i = 0; i < inputs.size(); i++) {
		auto input = inputs[i];
		auto gTree = goldTrees[i];
		int actNum = input.size() * 2; // n shift and n reduce
									   // one more reduce action for reduce root

		//get gold tree label, word, tag index
		//of a sentence string->int
		std::vector<int> labelIndexCache(gTree.size);
		std::vector<int> wordIndexCache(gTree.size);
		std::vector<int> tagIndexCache(gTree.size);

		int index = 0;
		for (auto iter = gTree.nodes.begin(); iter != gTree.nodes.end();
				iter++) {
			auto labelIdx = labelMap.find(iter->label);
			auto wordIdx = wordMap.find(iter->word);
			auto tagIdx = tagMap.find(iter->tag);

			if (labelIdx == labelMap.end()) {
				std::cerr << "Dep label " << iter->label
						<< " is not in labelMap!" << std::endl;
				exit(1);
			}

			if (wordIdx == wordMap.end()) {
				std::cerr << "Dep word " << iter->word << " is not in wordMap!"
						<< std::endl;
				exit(1);
			}

			if (tagIdx == tagMap.end()) {
				std::cerr << "Dep tag " << iter->tag << " is not in tagMap!"
						<< std::endl;
				exit(1);
			}
			labelIndexCache[index] = labelIdx->second;
			wordIndexCache[index] = wordIdx->second;
			tagIndexCache[index] = labelIdx->second;
			index++;
		}

		//get state features
		std::vector<int> acts(actNum); //gold acts
		std::vector<Example> examples; //features and labels

		State* state = new State();

		//for every state of a sentence
		for (int j = 0; !state->complete(); j++) {
			std::vector<int> features(featureNum);
			std::vector<int> labels(kActNum, 0);

			//get current state features
			featureExtract(state, wordIndexCache, tagIndexCache, features);

			//get current state valid actions
			state->getValidActs(labels);

			//find gold action and move to next
			int goldAct = state->StandardMove(gTree, labelIndexCache);
			acts[j] = goldAct;
			state->Move(goldAct);

			labels[goldAct] = 1;
			Example example(features, labels);
			examples.push_back(example);
		}

		gExamples[i].setParas(examples, acts, wordIndexCache, tagIndexCache);

		delete state;

	}

}

/**
 *   get the feature vector in all the beam states,
 *   and return the input layer of neural network in a batch.
 */
void Depparser::getInputBatch(State* state, std::vector<int>& wordIndexCache,
		std::vector<int>& tagIndexCache,
		std::vector<std::vector<int> >& featvecs) {

	for(int i = 0; i < featvecs.size(); i++){
		std::vector<int> featvec(CConfig::nFeatureNum);
		featureExtract( state + i, wordIndexCache, tagIndexCache, featvec);
		featvecs.push_back(featvec);
	}
}
