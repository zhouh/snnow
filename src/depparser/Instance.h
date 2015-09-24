/*************************************************************************
	> File Name: src/depparser/Instance.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 19/09/15 15:27:39
 ************************************************************************/
#ifndef DEPPARSER_INSTANCE_H
#define DEPPARSER_INSTANCE_H

#include "DepTree.h"
#include "FeatureExtractor.h"

class Instance{
    
public :
    DepParseInput input;
    std::vector<int> tagCache;
    std::vector<int> wordCache;

    Instance(DepParseInput input){
        
        this->input = input;
    }

    void getCache(FeatureExtractor& fe){

	    wordCache.resize(input.size());
	    tagCache.resize(input.size());

		int index = 0;
		for (auto iter = input.begin(); iter != input.end(); iter++) {

            int wordIdx = fe.getWord(iter->first);
            int tagIdx = fe.getTag(iter->second);


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

			wordCache[index] = wordIdx;
			tagCache[index] = tagIdx;
			index++;

        }
    }

    ~Instance(){
        
    }
        
};

#endif
