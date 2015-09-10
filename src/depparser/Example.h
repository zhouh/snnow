/*
 * Example.h
 *
 *  Created on: Jul 3, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_EXAMPLE_H_
#define DEPPARSER_EXAMPLE_H_

class Example {
public:
	std::vector<int> features;
	std::vector<int> labels;
	Example(std::vector<int> &f, std::vector<int> &l){
		features = f;
		labels = l;
	}
	virtual ~Example(){
	}
};

#endif /* DEPPARSER_EXAMPLE_H_ */
