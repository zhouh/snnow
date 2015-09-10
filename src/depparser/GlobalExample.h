/*
 * GlobalExample.h
 *
 *  Created on: Jul 3, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_GLOBALEXAMPLE_H_
#define DEPPARSER_GLOBALEXAMPLE_H_

#include "DepTree.h"
#include "Example.h"

class GlobalExample {
public:
	std::vector<Example> examples;
	std::vector<int> goldActs;
	std::vector<int> wordIdx;
	std::vector<int> tagIdx;

	GlobalExample(){}
	GlobalExample(std::vector<Example>& es, std::vector<int>& acts){
		examples = es;
		goldActs = acts;
	}
	~GlobalExample(){};

	inline void setParas(std::vector<Example> & es, std::vector<int>& acts,
			std::vector<int> & wIdx, std::vector<int> & tIdx){
		examples = es;
		goldActs = acts;
		wordIdx = wIdx;
		tagIdx = tIdx;

	}
};

#endif /* DEPPARSER_GLOBALEXAMPLE_H_ */
