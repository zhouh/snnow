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
    Instance instance;
    

    GlobalExample(std::vector<Example>& es, std::vector<int>& acts,
                  Instance & inst) : instance( inst) {
		examples = es;
		goldActs = acts;
	}
    ~GlobalExample(){
    };
};

#endif /* DEPPARSER_GLOBALEXAMPLE_H_ */
