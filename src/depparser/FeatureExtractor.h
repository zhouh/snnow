/*************************************************************************
	> File Name: FeatureExtractor.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/
#ifndef DEPPARSER_FEATUREEXTRATOR_H
#define DEPPARSER_FEATUREEXTRATOR_H

#include<iostream>
#include<tr1/unordered_map>
#include<unordered_set>

#include "DepTree.h"
#include "State.h"
#include "GlobalExample.h"

class FeatureExtractor{
    
private:
    std::vector<std::string> knowLabels;
	std::vector<std::string> knowWords;
	std::vector<std::string> knowTags;
	std::tr1::unordered_map<std::string, int> labelMap;
	std::tr1::unordered_map<std::string, int> wordMap;
	std::tr1::unordered_map<std::string, int> tagMap; 

    int wordNullIdx;
	int wordUnkIdx;
	int wordRootIdx;
	int tagNullIdx;
	int tagUnkIdx;
	int tagRootIdx;
	int labelNullIdx;

    public:
	const static int featureNum = 48;
    FeatureExtractor(){}

    inline void displayDict(){
        std::cout<< "knowLabels Size:" << knowLabels.size() << std::endl;
        std::cout<< "knowWords Size:" << knowWords.size() << std::endl;
        std::cout<< "knowTags Size:" << knowTags.size() << std::endl;
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

/**
 *   get the dictionary
 */
void getDictionaries(std::vector<DepTree> goldTrees); 

void featureExtract(State* state, std::vector<int>& wordIndexCache,
		std::vector<int>& tagIndexCache, std::vector<int> & features);
/**
 *   generate training examples for global learning
 *   assign the global example to gExamples, which is a member of class Depparser
 */
void generateTrainingExamples(std::vector<DepParseInput> inputs, std::vector<DepTree> goldTrees, 
                              std::vector<GlobalExample> & gExamples);

void getCache(Instance & inst){

    inst.input.wordIndexCache.resize(input.size());
    inst.input.tagIndexCache.resize(input.size());

	int index = 0;
	for (auto iter = inst.input.begin(); iter != inst.input.end(); iter++) {

        int wordIdx = getWord(iter->first);
        int tagIdx = getTag(iter->second);


		if ( wordIdx == -1 ) {
			std::cerr << "Dep word " << iter->first << " is not in wordMap!"
					<< std::endl;
			exit(1);
		}

		if ( tagIdx == -1 ) {
			std::cerr << "Dep tag " << iter->second << " is not in tagMap!"
					<< std::endl;
			exit(1);
		}

		inst.input.wordCache[index] = wordIdx;
		inst.input.tagCache[index] = tagIdx;
		index++;
    }
 }

};

#endif   /* END HEAD */
