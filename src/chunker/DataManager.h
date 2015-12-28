/*************************************************************************
	> File Name: DataManager.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 12:53:45 PM CST
 ************************************************************************/
#ifndef _CHUNKER_DATAMANAGER_H_
#define _CHUNKER_DATAMANAGER_H_

#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <unordered_set>
#include <memory>

#include "Instance.h"
#include "DictManager.h"
#include "ChunkedSentence.h"
#include "ActionStandardSystem.h"
#include "Example.h"

class DataManager{
public:
    std::tr1::unordered_map<std::string, std::shared_ptr<DictManager>> m_mStr2DictManager;

public:
    static const std::string WORDDESCRIPTION;
    static const std::string POSDESCRIPTION;
    // static const std::string LABELDESCRIPTION;
    static const std::string CAPDESCRIPTION;

public:
    DataManager() { }
    ~DataManager() {}

    void init(const ChunkedDataSet &goldSet) {
        m_mStr2DictManager[WORDDESCRIPTION] = std::shared_ptr<DictManager>(new WordDataManager());
        m_mStr2DictManager[POSDESCRIPTION] = std::shared_ptr<DictManager>(new POSDataManager());
        // m_mStr2DictManager[LABELDESCRIPTION] = std::shared_ptr<DictManager>(new LabelDataManager());
        m_mStr2DictManager[CAPDESCRIPTION] = std::shared_ptr<DictManager>(new CapitalDataManager());

        makeDictionaries(goldSet);
    }

    void makeDictionaries(const ChunkedDataSet &goldSet) {
        for (auto &it : m_mStr2DictManager) {
            it.second->makeDictionaries(goldSet);
        }
    }

    void generateInstanceCache(Instance &inst);

    void generateInstanceSetCache(InstanceSet &instSet) {
        for (auto &inst : instSet) {
            generateInstanceCache(inst);
        }
    }

    void generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples) {
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
            generateInstanceCache(inst);
    
            //generate every state of a sentence
            for (int j = 0; !state->complete(); j++) {
                FeatureVector features;
                std::vector<int> labels(transitionSystem.nActNum, 0);
    
                extractFeature(*state, inst, features);
    
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

private:
    DataManager(const DataManager &dManager)  = delete;
    DataManager& operator= (const DataManager &dManager) = delete;
};

#endif
