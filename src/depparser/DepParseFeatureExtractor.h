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

#include "mshadow/tensor.h"
#include "DepParseMacro.h"
#include "FeatureEmbedding.h"
#include "DataSet.h"
#include "DepParseDataSet.h"
#include "base/FeatureExtractor.h"
#include "base/TrainingExample.h"
#include "Dict.h"
#include "nets/Model.h"
#include "DepParseState.h"
#include "DepArcStandardSystem.h"

using namespace mshadow;

class DepParseFeatureExtractor : public FeatureExtractor{
    
public:

    // total feature num
	const static int c_featureNum = 48;

    // index of different dictionaries in the dict table
	const static int c_word_dict_index = 0;
	const static int c_tag_dict_index = 1;
	const static int c_dep_label_dict_index = 2;

    static std::string word_string;
    static std::string tag_string;
    static std::string label_string;

public:
    int feature_nums[3];

    // set of all feature dictionary
    DictionaryVectorPtrs dictionary_ptrs_table;
    FeatureTypes feature_types;

public:

    DepParseFeatureExtractor() {
        feature_nums[0] = 18;
        feature_nums[1] = 18;
        feature_nums[2] = 12;
    }

    void setFeatureTypes(FeatureTypes ft) {
        this->feature_types = ft;
    }

    inline int getTotalInputSize(){

        int retval = 0;
        for (int i = 0; i < feature_types.size(); ++i) {
            retval += feature_types[i].feature_embedding_size * feature_nums[i];

        }
        return retval;
    }

    /**
     * return the know label set
     */
    const std::vector<std::string>& getKnownDepLabelVector() {
        return dictionary_ptrs_table[c_dep_label_dict_index]->getKnownStringVector();
    }
    const String2IndexMap& getKnownDepLabelVectorMap() {
        return dictionary_ptrs_table[c_dep_label_dict_index]->getMap();
    }

    inline void displayDict(){
        std::clog<< "###knowLabels Size:" << dictionary_ptrs_table[c_dep_label_dict_index]->size() << std::endl;
        std::clog<< "###knowWords Size:" << dictionary_ptrs_table[c_word_dict_index]->size() << std::endl;
        std::clog<< "###knowTags Size:" << dictionary_ptrs_table[c_tag_dict_index]->size() << std::endl;
    }

    /**
     * directly get the index from the dictionary
     */
    inline int getWordIndex(std::string& s) {
		return dictionary_ptrs_table[c_word_dict_index]->getStringIndex(s);
	}

	inline int getTagIndex(std::string& s) {
        return dictionary_ptrs_table[c_tag_dict_index]->getStringIndex(s);

	}

	inline int getLabelIndex(std::string& s) {
        return dictionary_ptrs_table[c_dep_label_dict_index]->getStringIndex(s);
	}
    //=====================================================================


    /**
     *  @index the index in the sentence
     *  @list cache of dictionary index for the sentence
     */
    inline int getWordIndex(int index, std::vector<int> & list){

        if(index == -1) {
            return dictionary_ptrs_table[c_word_dict_index]->getNullIndex();
        }

        return list[index];
    }

    /**
     *  @index the index in the sentence
     *  @list cache of dictionary index for the sentence
     */
	inline int getTagIndex(int index, std::vector<int> & list){
		if(index == -1)
			return dictionary_ptrs_table[c_tag_dict_index]->getNullIndex();
		return list[index];
	}

    /**
     *  @index the index of query label in the tree
     *  @DepParseState cache of dep label is maintained by the state
     */
	inline int getLabelIndex(int index, DepParseState* state){
		if(index == -1)
			return dictionary_ptrs_table[c_dep_label_dict_index]->getNullIndex();
		return state->label(index);
	}
    //================================================================

    /**
     *   get the dictionary
     */
    void getDictionaries(DataSet& data);

    std::shared_ptr<FeatureVector> getFeatureVectors(const State& state, const Input& input);

//    /**
//     *   generate training examples for global learning
//     *   assign the global example to gExamples, which is a member of class Depparser
//     */
//    void generateTrainingExamples(ArcStandardSystem * tranSystem, std::vector<Instance> & instances,
//            std::vector<DepTree> & goldTrees, std::vector<GlobalExample> & gExamples);
	void generateGreedyTrainingExamples(DepArcStandardSystem* transit_system, DepParseDataSet& training_data, std::vector<std::shared_ptr<Example>> & examples);

    /**
     * get cache for the input
     */
    void getCache(DepParseInput& input){

        // resize the cache
        input.word_cache.resize(input.size());
        input.tag_cache.resize(input.size());
    
    	int index = 0;
    	for (auto iter = input.begin(); iter != input.end(); iter++) {
    
            int word_idx = getWordIndex(iter->first);
            int tag_idx = getTagIndex(iter->second);
    
    		if ( word_idx == -1 ) {
    			std::cerr << "Dep word " << iter->first << " is not in wordMap!"
    					<< std::endl;
    			exit(1);
    		}
    
    		if ( tag_idx == -1 ) {
    			std::cerr << "Dep tag " << iter->second << " is not in tagMap!"
    					<< std::endl;
    			exit(1);
    		}

            input.word_cache[index] = word_idx;
            input.tag_cache[index] = tag_idx;
    		index++;
        }
    }

    /**
     * get cache for the inputs
     */
    void getInputsCache(std::vector<DepParseInput> & inputs){
        for(auto & inst : inputs)
            getCache(inst);
    }

    void readPretrainedEmbeddings(Model<cpu> & model, std::string file_name) {
        for (int i = 0; i < static_cast<int>(feature_types.size()); i++) {
            if (feature_types[i].type_name == DepParseFeatureExtractor::word_string) {
                model.featEmbs[i]->readPreTrain(file_name, dictionary_ptrs_table[i]);
            }
        }
    }

    void returnInput(std::vector<FeatureVector> &featVecs,
                                              std::vector<std::shared_ptr<FeatureEmbedding>> &featEmbs,
                                              TensorContainer<cpu, 2, real_t> & input){
        for(unsigned beamIndex = 0; beamIndex < static_cast<unsigned>(featVecs.size()); beamIndex++) { // for every beam item

            FeatureVector &featVector = featVecs[beamIndex];

            int inputIndex = 0;
            for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featVector.size()); featTypeIndex++) {
                const std::vector<int> &curFeatVec = featVector[featTypeIndex];
                const int curFeatSize = feature_types[featTypeIndex].feature_size;
                const int curEmbSize  = feature_types[featTypeIndex].feature_embedding_size;
                std::shared_ptr<FeatureEmbedding> &curFeatEmb = featEmbs[featTypeIndex];

                for (auto featId : curFeatVec) {
                    Copy(input[beamIndex].Slice(inputIndex, inputIndex + curEmbSize), curFeatEmb->data[featId], curFeatEmb->data.stream_);
                    inputIndex += curEmbSize;
                }
            }
        }
    }

};

#endif   /* END HEAD */
