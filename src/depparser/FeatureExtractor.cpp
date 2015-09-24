/*************************************************************************
	> File Name: src/depparser/FeatureExtractor.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include<iostream>

#include "FeatureExtractor.h"

/**
 *   get the dictionary
 */
void FeatureExtractor::getDictionaries(std::vector<DepTree> goldTrees) {

	std::unordered_set<std::string> labelSet;
	std::unordered_set<std::string> tagSet;
	std::unordered_set<std::string> wordSet;

	// insert to the set
	for (unsigned i = 0; i < goldTrees.size(); i++) {
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

void FeatureExtractor::featureExtract(State* state, std::vector<int>& wordIndexCache,
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

#ifdef DEBUG
    std::cout<<"s0:"<<s0<<std::endl;
#endif
	s0l = s0 == -1 ? -1 : state->leftdep(s0);
	s0r = s0 == -1 ? -1 : state->rightdep(s0);
	s0l2 = s0 == -1 ? -1 : state->left2dep(s0);
	s0r2 = s0 == -1 ? -1 : state->left2dep(s0);
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
	s1l = s1 == -1 ? -1 : state->leftdep(s1);
	s1r = s1 == -1 ? -1 : state->rightdep(s1);
	s1l2 = s1 == -1 ? -1 : state->left2dep(s1);
	s1r2 = s1 == -1 ? -1 : state->left2dep(s1);
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
 *   assign the global example to gExamples, which is a member of class Depparser
 */
void FeatureExtractor::generateTrainingExamples(std::vector<DepParseInput> inputs,
		std::vector<DepTree> goldTrees, std::vector<GlobalExample>& gExamples) {

    gExamples.clear();

	//for every sentence, cache the word and tag hash index
	for (unsigned i = 0; i < inputs.size(); i++) {
		auto input = inputs[i];
		auto gTree = goldTrees[i];
		int actNum = input.size() * 2; // n shift and n reduce
									   // one more reduce action for root

        /*
         * cache the dependency label in the training set
         */
		std::vector<int> labelIndexCache(gTree.size);
		int index = 0;
		for (auto iter = gTree.nodes.begin(); iter != gTree.nodes.end();
				iter++) {
			auto labelIdx = labelMap.find(iter->label);

			if (labelIdx == labelMap.end()) {
				std::cerr << "Dep label " << iter->label
						<< " is not in labelMap!" << std::endl;
				exit(1);
			}

			labelIndexCache[index] = labelIdx->second;
			index++;
		}

		//get state features
		std::vector<int> acts(actNum); //gold acts
		std::vector<Example> examples; //features and labels

		State* state = new State();
        state->len_ = inputs[i].size();

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

#ifdef DEBUG
            std::cout << "move action:  "<< goldAct;
            std::cout << DecodeUnlabeledAction(goldAct)<<std::endl;
#endif
			state->Move(goldAct);

			labels[goldAct] = 1;
			Example example(features, labels);
			examples.push_back(example);
		}

        GlobalExample ge_i( examples, acts, input );
        getCache(ge_i.instance);

#ifdef DEBUG
        std::cout << "========================================"<< std::endl;
#endif

		gExamples.push_back(ge_i);
		delete state;
	}
}


