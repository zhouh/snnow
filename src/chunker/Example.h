/*************************************************************************
	> File Name: Example.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 08:27:33 PM CST
 ************************************************************************/
#ifndef _CHUNKER_EXAMPLE_H_
#define _CHUNKER_EXAMPLE_H_

#include <vector>

#include "Instance.h"
#include "FeatureVector.h"

class Example {
public:
    FeatureVector features;
    std::vector<int> labels;

    Example(FeatureVector &f, std::vector<int> &l) : features(f), labels(l){

    }
    Example(const Example &e) : features(e.features), labels(e.labels) {}
    Example& operator=(const Example &e) {
        if (this == &e) {
            return *this;
        }

        this->features = e.features;
        this->labels = e.labels;

        return *this;
    }

    ~Example() {}
};

class GlobalExample {
public:
    std::vector<Example> examples;
    std::vector<int> goldActs;
    Instance instance;

    GlobalExample(std::vector<Example> &exs, std::vector<int> &gActs, Instance &inst): examples(exs), goldActs(gActs), instance(inst) {}
    ~GlobalExample() {}
};

typedef std::vector<GlobalExample> GlobalExamples;

#endif
