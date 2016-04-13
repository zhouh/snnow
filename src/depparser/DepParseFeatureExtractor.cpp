/*************************************************************************
	> File Name: src/depparser/DepParseFeatureExtractor.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include<iostream>
#include<algorithm>
#include<cctype>

#include "DepParseFeatureExtractor.h"

/**
 *   prepare the dictionary for feature extracting.
 *   The index of the dic here are independent
 */
void DepParseFeatureExtractor::getDictionaries(DataSet& data) {

	data = static_cast<DepParseDataSet>(data);

	StringSet labelSet;
	StringSet tagSet;
	StringSet wordSet;

	// insert to the set
    int data_size = data.getSize();
	for (unsigned i = 0; i < data_size; i++) {
        DepParseTree gold_tree_i = static_cast<DepParseTree>(data.outputs[i]);

		for (int j = 1; j < gold_tree_i->size; j++) { // the first node is -ROOT- node, skip
			auto tree_node = gold_tree_i->nodes[j];
			labelSet.insert(tree_node.label);
			tagSet.insert(tree_node.tag);
			wordSet.insert(tree_node.word);
		}
	}

	// insert into the map
	int index = c_dictionary_begin_index;
	//label
    for (auto it = labelSet.begin(); it != labelSet.end(); ++it) {
        labelMap[*it] = index++;
        know_dependency_labels.insert(*it);  // prepare the know labels for build the transition system of parsing
    }

    int index = c_dictionary_begin_index;
    for (auto it = tagSet.begin(); it != tagSet.end(); ++it) {
        tagMap[*it] = index++;
    }

    int index = c_dictionary_begin_index;
    for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
		wordMap[*it] = index++;
	}
}

void DepParseFeatureExtractor::featureExtract(State* state, std::vector<int>& wordIndexCache,
		std::vector<int>& tagIndexCache, std::vector<int> & features) {
	// positions 0-17 hold fWord, 18-35 hold fPos, 36-47 hold fLabel
//	int POS_OFFSET = 18;
//	int DEP_OFFSET = 36;
//	int STACK_OFFSET = 6;
//	int STACK_NUMBER = 6;

//	int index = 0;
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
    // 0 - 12
	features[IDIdx++] = getWordIndex(s0, wordIndexCache); // 0
	features[IDIdx++] = getWordIndex(s1, wordIndexCache); // 1
	features[IDIdx++] = getWordIndex(s2, wordIndexCache); // 2
	features[IDIdx++] = getWordIndex(q0, wordIndexCache); // 3
	features[IDIdx++] = getWordIndex(q1, wordIndexCache); // 4
	features[IDIdx++] = getWordIndex(q2, wordIndexCache); // 5

	features[IDIdx++] = getTagIndex(s0, tagIndexCache);   // 6
	features[IDIdx++] = getTagIndex(s1, tagIndexCache);   // 7
	features[IDIdx++] = getTagIndex(s2, tagIndexCache);   // 8
	features[IDIdx++] = getTagIndex(q0, tagIndexCache);   // 9
	features[IDIdx++] = getTagIndex(q1, tagIndexCache);   // 10
	features[IDIdx++] = getTagIndex(q2, tagIndexCache);   // 11

	// s0l s0r s0l2 s0r2 s0ll s0rr
	int s0l, s0r, s0l2, s0r2, s0ll, s0rr;

#ifdef DEBUG
    std::cout<<"s0:"<<s0<<std::endl;
#endif
	s0l = s0 == -1 ? -1 : state->leftdep(s0);
	s0r = s0 == -1 ? -1 : state->rightdep(s0);
	s0l2 = s0 == -1 ? -1 : state->left2dep(s0);
	s0r2 = s0 == -1 ? -1 : state->left2dep(s0);
	s0ll = s0l == -1 ? -1 : state->leftdep(s0l);
	s0rr = s0r == -1 ? -1 : state->rightdep(s0r);
    
    //12 - 29
	features[IDIdx++] = getWordIndex(s0l, wordIndexCache);   // 12
	features[IDIdx++] = getWordIndex(s0r, wordIndexCache);   // 13
	features[IDIdx++] = getWordIndex(s0l2, wordIndexCache);  // 14
	features[IDIdx++] = getWordIndex(s0r2, wordIndexCache);  // 15
	features[IDIdx++] = getWordIndex(s0ll, wordIndexCache);  // 16
	features[IDIdx++] = getWordIndex(s0rr, wordIndexCache);  // 17

	features[IDIdx++] = getTagIndex(s0l, tagIndexCache);     // 18   
	features[IDIdx++] = getTagIndex(s0r, tagIndexCache);     // 19
	features[IDIdx++] = getTagIndex(s0l2, tagIndexCache);    // 20
	features[IDIdx++] = getTagIndex(s0r2, tagIndexCache);    // 21
	features[IDIdx++] = getTagIndex(s0ll, tagIndexCache);    // 22
	features[IDIdx++] = getTagIndex(s0rr, tagIndexCache);    // 23
    //std::cout<<IDIdx<<" "<<s0rr<<" "<<getTagIndex(s0rr, tagIndexCache)<<std::endl;

	features[IDIdx++] = getLabelIndex(s0l, state);  // 24
	features[IDIdx++] = getLabelIndex(s0r, state);  // 25
	features[IDIdx++] = getLabelIndex(s0l2, state); // 26
	features[IDIdx++] = getLabelIndex(s0r2, state); // 27
	features[IDIdx++] = getLabelIndex(s0ll, state); // 28
	features[IDIdx++] = getLabelIndex(s0rr, state); // 29

	// s1l s1r s1l2 s1r2 s1ll s1rr
	int s1l, s1r, s1l2, s1r2, s1ll, s1rr;
	s1l = s1 == -1 ? -1 : state->leftdep(s1);
	s1r = s1 == -1 ? -1 : state->rightdep(s1);
	s1l2 = s1 == -1 ? -1 : state->left2dep(s1);
	s1r2 = s1 == -1 ? -1 : state->left2dep(s1);
	s1ll = s1l == -1 ? -1 : state->leftdep(s1l);
	s1rr = s1r == -1 ? -1 : state->rightdep(s1r);

    // 30 - 47
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
 * generate the training examples for the greedy
 */
void DepParseFeatureExtractor::generateGreedyTrainingExamples(
        DepArcStandardSystem* transit_system_ptr,
        DepParseDataSet& training_data,
        std::vector<std::shared_ptr<Example>>& examples){

    examples.clear();

    //for every sentence, cache the word and tag hash index
    for (unsigned i = 0; i < training_data.getSize(); i++) {

        auto trees = training_data.outputs;
        trees = static_cast<std::vector<std::shared_ptr<DepParseTree>> >(trees);
        auto inputs = training_data.inputs;
        inputs = static_cast<std::vector<std::shared_ptr<DepParseInput>> >(inputs);



        Instance & inst = instances[i];
        auto & input_ptr_i = inputs[i];
        auto & tree_ptr_i = trees[i];
        int actNum = ( input.size() - 1 ) * 2; // n shift and n reduce
        // one more reduce action for root

        /*
         * cache the dependency label in the training set
         */
        std::vector<int> labelIndexCache(gTree.size);
        int index = 0;
        for (auto iter = gTree.nodes.begin(); iter != gTree.nodes.end();
             iter++) {
            int labelIndex = getLabel(iter->label);

            if (labelIndex == -1) {
                std::cerr << "Dep label " << iter->label
                << " is not in labelMap!" << std::endl;
                exit(1);
            }

            labelIndexCache[index] = labelIndex;
            index++;
        }

        //get state features
        std::vector<int> acts(actNum); //gold acts
        std::vector<Example> examples; //features and labels

        State* state = new State();
        state->len_ = input.size();
        state->initCache();
        getCache(inst);

        //for every state of a sentence
        for (int j = 0; !state->complete(); j++) {
            std::vector<int> features(featureNum);
            std::vector<int> labels(tranSystem->nActNum, 0);

            //get current state features
            featureExtract(state, inst.wordCache, inst.tagCache, features);

            //get current state valid actions
            tranSystem->getValidActs(*state, labels);

            //find gold action and move to next
            int goldAct = tranSystem->StandardMove(*state, gTree, labelIndexCache);
            acts[j] = goldAct;

            tranSystem->Move(*state, goldAct);
/*            state->display();*/
            //state->dispalyCache();

            labels[goldAct] = 1;
            Example example(features, labels);
            examples.push_back(example);
        }

        GlobalExample ge_i( examples, acts, inst );

        gExamples.push_back(ge_i);

        delete state;

    }

}


/**
 *   generate training examples for global learning
 *   assign the global example to gExamples, which is a member of class Depparser
 */
void DepParseFeatureExtractor::generateTrainingExamples( ArcStandardSystem * tranSystem, std::vector<Instance>& instances,
		std::vector<DepTree>& goldTrees, std::vector<GlobalExample>& gExamples ) {

    gExamples.clear();

	//for every sentence, cache the word and tag hash index
	for (unsigned i = 0; i < instances.size(); i++) {

        Instance & inst = instances[i];
		auto & input = inst.input;
		auto & gTree = goldTrees[i];
		int actNum = ( input.size() - 1 ) * 2; // n shift and n reduce
									   // one more reduce action for root

        /*
         * cache the dependency label in the training set
         */
		std::vector<int> labelIndexCache(gTree.size);
		int index = 0;
		for (auto iter = gTree.nodes.begin(); iter != gTree.nodes.end();
				iter++) {
            int labelIndex = getLabel(iter->label);

			if (labelIndex == -1) {
				std::cerr << "Dep label " << iter->label
						<< " is not in labelMap!" << std::endl;
				exit(1);
			}

			labelIndexCache[index] = labelIndex;
			index++;
		}

		//get state features
		std::vector<int> acts(actNum); //gold acts
		std::vector<Example> examples; //features and labels

		State* state = new State();
        state->len_ = input.size();
        state->initCache();
        getCache(inst);
        
		//for every state of a sentence
		for (int j = 0; !state->complete(); j++) {
			std::vector<int> features(featureNum);
			std::vector<int> labels(tranSystem->nActNum, 0);

			//get current state features
			featureExtract(state, inst.wordCache, inst.tagCache, features);

			//get current state valid actions
			tranSystem->getValidActs(*state, labels);

            /*
             * display valid actions
             */
            /*for(int va = 0; va < labels.size(); va++){*/
                //if( labels[va] < 0 )
                    //continue;
                //if( va == tranSystem->nLeftFirst || va == tranSystem->nRightFirst )
                    //std::cout<<std::endl;

                //std::cout<<tranSystem->DecodeUnlabeledAction(va)<<"+"<<tranSystem->DecodeLabel(va)<<" ";

            //}

			//find gold action and move to next
			int goldAct = tranSystem->StandardMove(*state, gTree, labelIndexCache);
			acts[j] = goldAct;

            //std::cout << "move action:  "<< goldAct<<":"<<tranSystem->DecodeUnlabeledAction(goldAct)<<"+"<<tranSystem->DecodeLabel(goldAct) << std::endl;
			tranSystem->Move(*state, goldAct);
/*            state->display();*/
            //state->dispalyCache();

			labels[goldAct] = 1;
			Example example(features, labels);
			examples.push_back(example);
		}

        GlobalExample ge_i( examples, acts, inst );

		gExamples.push_back(ge_i);

		delete state;

#ifdef DEBUG
        std::cout << "========================================"<< std::endl;
#endif

	}
}


