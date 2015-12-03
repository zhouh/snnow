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

class Example {
public:
    std::vector<int> featuers;
    std::vector<int> labels;

    Example(std::vector<int> &f, std::vector<int> &l){
        featuers = f;
        labels = l;
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
