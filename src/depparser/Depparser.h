/*
 * Depparser.h
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_DEPPARSER_H_
#define DEPPARSER_DEPPARSER_H_

#include <algorithm>

#include "Dict.h"
#include "DepTree.h"
#include "State.h"


class Depparser {

private:
	std::vector<std::string> knowLabels;
	std::vector<std::string> knowWords;
	std::vector<std::string> knowTags;
	std::unordered_map<std::string, int> labelMap;
	std::unordered_map<std::string, int> wordMap;
	std::unordered_map<std::string, int> tagMap;

	std::vector<GlobalExample> gExamples;

	int beamSize;
	bool m_bTrain;
	bool bEarlyUpdate = true;

	static int featureNum = 48;
	static int wordNullIdx;
	static int wordUnkIdx;
	static int wordRootIdx;
	static int tagNullIdx;
	static int tagUnkIdx;
	static int tagRootIdx;
	static int labelNullIdx;


	const int labelNum;
	Dict depLabels;

public:
	Depparser(bool bTrain);
	Depparser();
	virtual ~Depparser();

	void init(std::vector<DepParseInput> inputs,
			std::vector<DepTree> goldTrees);
	/**
	 * train the input sentences with mini-batch adaGrad
	 */
	void train(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees,
			std::vector<DepParseInput> devInputs, std::vector<DepTree> devTrees);

	void parse(std::vector<DepParseInput> inputs);

	void featureExtract(State* state, std::vector<int>& wordIndexCache,
			std::vector<int>& tagIndexCache, std::vector<int>& features);

private:
	void getDictionaries(std::vector<DepTree> goldTrees);

	void generateTrainingExamples(std::vector<DepParseInput> inputs,
			std::vector<DepTree> goldTrees);

	void getInputBatch(State* state, std::vector<int>& wordIndexCache,
			std::vector<int>& tagIndexCache, std::vector< std::vector<int> >& features);

	/**
	 *  Insert one transition into beam,
	 *  if beam is full, pop and push, return 0;
	 *  if beam is not full, push directly, return 1.
	 */
	inline int insertBeam(const CScoredTransition& transition, CScoredTransition* beamTransitsint,
			int currentBeamSize = beamSize){
		//beam is full
		if (currentBeamSize == beamSize) {
		    if (transition.score > beamTransitsint[0].score) {
		      std::pop_heap(beamTransitsint, beamTransitsint + beamSize,
		          ScoredTransitionMore);
		      beamTransitsint[beamSize- 1] = transition;
		      std::push_heap(beamTransitsint, beamTransitsint+ beamSize,
		          ScoredTransitionMore);
		    }
		    return 0;
		  }

		//beam not full, insert directly
		beamTransitsint[currentBeamSize] = transition;
		std::push_heap(currentBeamSize,
				currentBeamSize + currentBeamSize + 1,
				ScoredTransitionMore);
		++ currentBeamSize;
		return 1;
	}

	inline int getWord(std::string s) {
		auto got = wordMap.find(s);
		return got == wordMap.end() ? -1 : got->second;
	}
	inline int getTag(std::string s) {
		auto got = tagMap.find(s);
		return got == tagMap.end() ? -1 : got->second;
	}
	inline int getLabel(std::string s) {
		auto got = labelMap.find(s);
		return got == labelMap.end() ? -1 : got->second;
	}

	inline int getWordIndex(int index, std::vector<int> list){

		if(index == -1)
			return wordNullIdx;

		return list[index];
	}

	inline int getTagIndex(int index, std::vector<int> list){
		if(index == -1)
			return tagNullIdx;
		return list[index];
	}

	inline int getLabelIndex(int index, State* state){
		if(index == -1)
			return labelNullIdx;
		return state->label(index);
	}

};

#endif /* DEPPARSER_DEPPARSER_H_ */
