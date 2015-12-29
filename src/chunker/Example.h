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
#include "ActionStandardSystem.h"
#include "FeatureManager.h"
#include "DictManager.h"
#include "ChunkedSentence.h"

class GlobalExample;

typedef std::vector<GlobalExample> GlobalExamples;

class Example {
public:
    FeatureVector features;
    std::vector<int> labels;

    Example(const FeatureVector &f, const std::vector<int> &l) : features(f), labels(l){
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

    static void generateTrainingExamples(ActionStandardSystem &transitionSystem, DictManager &dictManager, FeatureManager &featManager, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples) { 
        gExamples.clear();
    
        for (int i = 0; i < static_cast<int>(instSet.size()); i++) {
            Instance &inst = instSet[i];
            ChunkedSentence &gSent = goldSet[i];
        
            ChunkerInput &input = inst.input;
            int actNum = input.size();
    
            std::vector<int> labelIndexCache(gSent.size());
            int index = 0;
            for (const ChunkedWord &w : gSent.getChunkedWords()) {
                int labelIdx = transitionSystem.labelManager.label2Idx(w.label);
    
                labelIndexCache[index] = labelIdx;
                index++;
            }
    
            std::vector<int> acts(actNum);
            std::vector<Example> examples;
    
            std::shared_ptr<State> state(new State());
            state->m_nLen = input.size();
            Instance::generateInstanceCache(dictManager, inst);
    
            //generate every state of a sentence
            for (int j = 0; !state->complete(); j++) {
                FeatureVector features;
                std::vector<int> labels(transitionSystem.nActNum, 0);
    
                featManager.extractFeature(*state, inst, features);
    
                transitionSystem.generateValidActs(*state, labels);
    
                int goldAct = transitionSystem.standardMove(*state, gSent, labelIndexCache);
                acts[j] = goldAct;
    
                CScoredTransition tTrans(NULL, goldAct, 0);
                transitionSystem.move(*state, *state, tTrans);
    
                labels[goldAct] = 1;
    
                Example example(features, labels);
                examples.push_back(example);
            }
            GlobalExample tGlobalExampe(examples, acts, inst);
            gExamples.push_back(tGlobalExampe);
        }
    }

};


#endif
