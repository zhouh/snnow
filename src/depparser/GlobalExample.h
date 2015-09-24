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
#include "Instance.h"

class GlobalExample {
public:
	std::vector<Example> examples;
	std::vector<int> goldActs;
    Instance * instance;
    

	GlobalExample(){}
	GlobalExample(std::vector<Example>& es, std::vector<int>& acts,
                  DepParseInput input) {
		examples = es;
		goldActs = acts;
        instance = new Instance( input );
	}
    ~GlobalExample(){
        delete instance;
    };
};

#endif /* DEPPARSER_GLOBALEXAMPLE_H_ */
