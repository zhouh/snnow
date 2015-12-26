/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "FeatureManager.h"

void FeatureManager::init(const ChunkedDataSet &goldSet, double initRange, const bool readPretrainEmbs, const std::string &pretrainFile ) {
    int dicSize = 0;
    int featSize = 0;
    int featEmbSize = 0;

    totalFeatSize = 0;

    dataManager.makeDictionaries(goldSet);

    std::string wordFeatDescription = ChunkerDataManager::WORDDESCRIPTION;
    std::shared_ptr<DataManager> wordDataManager = dataManager.m_mDesc2dataManager[wordFeatDescription];
    dicSize = wordDataManager->size();
    featSize = 5;
    featEmbSize = 50;
    FeatureType wordFeatType(wordFeatDescription, featSize, featEmbSize);
    FeatureEmbedding *wordFeatEmb = new FeatureEmbedding(dicSize, featEmbSize, initRange);
    if (readPretrainEmbs) {
        readPretrainedEmbeddings(pretrainFile, wordDataManager->m_mElement2Idx, wordFeatEmb);
    }
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new WordFeatureExtractor(
                wordFeatType,
                wordDataManager,
                wordFeatEmb
                )));
    totalFeatSize += featSize * featEmbSize;

    std::string capFeatDescription = ChunkerDataManager::CAPDESCRIPTION;
    std::shared_ptr<DataManager> capDataManager = dataManager.m_mDesc2dataManager[capFeatDescription];
    dicSize = capDataManager->size();
    featSize = 1;
    featEmbSize = 5;
    FeatureType capFeatType(capFeatDescription, featSize, featEmbSize);
    FeatureEmbedding *capFeatEmb = new FeatureEmbedding(dicSize, featEmbSize, initRange);
    featExtractorPtrs.push_back(std::shared_ptr<FeatureExtractor>(new CapitalFeatureExtractor(
                capFeatType,
                capDataManager,
                capFeatEmb
                )));
    totalFeatSize += featSize * featEmbSize;
}

void FeatureManager::extractFeature(State &state, Instance &inst, FeatureVector &featVec) {
    for (auto &fe : featExtractorPtrs) {
        featVec.push_back(fe->extract(state, inst));
    }
}

int FeatureManager::readPretrainedEmbeddings(const std::string &pretrainFile, const std::tr1::unordered_map<std::string, int> &word2IdxMap, FeatureEmbedding *fEmb) {
    std::tr1::unordered_map<std::string, int> pretrainWords;
    std::vector<std::vector<double>> pretrainEmbs;
    std::string line;
    std::ifstream in(pretrainFile);
    getline(in, line); //TODO dirrent from zhouh
 
    int index = 0;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
 
        std::istringstream iss(line);
        std::vector<double> embedding;
 
        std::string word;
        double d;
        iss >> word;
        while (iss >> d) {
            embedding.push_back(d);
        }
 
        pretrainEmbs.push_back(embedding);
        pretrainWords[word] = index++;
    }
 
#ifdef DEBUG
    std::cerr << "  pretrainWords's size: " << pretrainEmbs.size() << std::endl;
#endif

    for (auto &wordPair : word2IdxMap) {
        auto ret = pretrainWords.find(wordPair.first);

        if (pretrainWords.end() != ret) {
            fEmb->getPreTrain(wordPair.second, pretrainEmbs[ret->second]);
        }
    }

    return static_cast<int>(pretrainWords.size());
}

void FeatureManager::generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples) {
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
        dataManager.generateInstanceCache(inst);

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

void FeatureManager::returnInput(std::vector<FeatureVector> &featVecs, TensorContainer<cpu, 2, double>& input, int beamSize){
	// initialize the input
	input.Resize( Shape2( beamSize, totalFeatSize ), 0.0);

	for(unsigned beamIndex = 0; beamIndex < featVecs.size(); beamIndex++) { // for every beam item
        FeatureVector &featVector = featVecs[beamIndex];

		int inputIndex = 0;
        for (int featTypeIndex = 0; featTypeIndex < static_cast<int>(featExtractorPtrs.size()); featTypeIndex++) {
            const std::vector<int> &curFeatVec = featVector[featTypeIndex];
            const int curFeatSize = featExtractorPtrs[featTypeIndex]->featType.featSize;
            const int curEmbSize  = featExtractorPtrs[featTypeIndex]->featType.featEmbSize;
            FeatureEmbedding &curFeatEmb = *(featExtractorPtrs[featTypeIndex]->fEmbPtr.get());

            for (int featureIndex = 0; featureIndex < curFeatSize; featureIndex++) {
                for (int embIndex = 0; embIndex < curEmbSize; embIndex++) {
                    input[beamIndex][inputIndex++] = curFeatEmb[curFeatVec[featureIndex]][embIndex];
                }
            }
        }
    }
}
