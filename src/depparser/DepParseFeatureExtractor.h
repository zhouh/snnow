/*************************************************************************
	> File Name: DepParseFeatureExtractor.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/
#ifndef SNNOW_DEPPARSE_FEATUREEXTRATOR_H
#define SNNOW_DEPPARSE_FEATUREEXTRATOR_H

#include<iostream>
#include<sstream>
#include<string>
#include<tr1/unordered_map>
#include<unordered_set>

#include "DepParseMacro.h"
#include "FeatureEmbedding.h"
#include "DataSet.h"
#include "DepParseDataSet.h"
#include "base/FeatureExtractor.h"
#include "base/TrainingExample.h"

class DepParseFeatureExtractor : public FeatureExtractor{
    
public:
	StringVector labelMap;
	StringVector wordMap;
	StringVector tagMap;

    StringSet know_dependency_labels;  // prepare the know labels for build the transition system of parsing


	const static int featureNum = 48;

public:
    DepParseFeatureExtractor(){}

    /**
     * return the know label set
     */
    StringSet getKnownDepLabelSet() {
        return know_dependency_labels;
    }

    /*
     * return the whole size of feature dictionary
     */
    inline int getDicSize(){
        return wordMap.size() + tagMap.size() + labelMap.size(); 
    }

    inline void displayDict(){
        std::clog<< "knowLabels Size:" << labelMap.size() << std::endl;
        std::clog<< "knowWords Size:" << wordMap.size() << std::endl;
        std::clog<< "knowTags Size:" << tagMap.size() << std::endl;
    }

    /**
     * getting index function, for checking whether the looking up is a unk
     */
    inline int getWord(std::string s) {
		auto got = wordMap.find(s);
		return got == wordMap.end() ? c_unk_index : got->second;
	}

	inline int getTag(std::string s) {
		auto got = tagMap.find(s);
		return got == tagMap.end() ? c_unk_index : got->second;
	}

	inline int getLabel(std::string s) {
		auto got = labelMap.find(s);

        if( got == labelMap.end() ){
            std::cerr<<"dep label not found : "<<s<<std::endl;
            exit(0);
        }
        return got->second;
	}
    //=====================================================================


    inline int getWordIndex(int index, std::vector<int> & list){

        if(index == -1) {
            return c_null_index;
        }

        if(index >= list.size()){
            std::cerr<<"in getWordIndex, the index out ot label set size! index : "<< index<<std::endl;
            exit(0);
        }
        return list[index];
    }


	inline int getTagIndex(int index, std::vector<int> & list){
		if(index == -1)
			return c_null_index;
		return list[index];
	}

	inline int getLabelIndex(int index, State* state){
		if(index == -1)
			return c_null_index;
		return state->label(index);
	}

    /**
     *   get the dictionary
     */
    void getDictionaries(DataSet& data);
    
    void featureExtract(State* state, std::vector<int>& wordIndexCache,
    		std::vector<int>& tagIndexCache, std::vector<int> & features);
    /**
     *   generate training examples for global learning
     *   assign the global example to gExamples, which is a member of class Depparser
     */
    void generateTrainingExamples(ArcStandardSystem * tranSystem, std::vector<Instance> & instances,
            std::vector<DepTree> & goldTrees, std::vector<GlobalExample> & gExamples);

	void generateGreedyTrainingExamples(DepArcStandardSystem* transit_system, DepParseDataSet& training_data, std::vector<std::shared_ptr<Example>> & examples);


    void getCache(DepParseInput& input){
    
        inst.wordCache.resize(inst.input.size());
        inst.tagCache.resize(inst.input.size());
    
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
    
    		inst.wordCache[index] = wordIdx;
    		inst.tagCache[index] = tagIdx;
    		index++;
        }
    }

    void getInstancesCache(std::vector<Instance> & insts){
        for(auto & inst : insts)
            getCache(inst);
    }

};

#endif   /* END HEAD */
