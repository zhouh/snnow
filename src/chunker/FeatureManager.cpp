/*************************************************************************
	> File Name: FeatureManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:40:03 PM CST
 ************************************************************************/
#include <memory>

#include "FeatureManager.h"

void FeatureManager::init(const ChunkedDataSet &goldSet, double initRange) {
    labelFeature = new LabelFeature();
    labelFeature->getDictionaries(goldSet);

    posFeature = new POSFeature();
    posFeature->getDictionaries(goldSet);

    int dicSize = 0;
    int featureSize = 0;
    int embeddingSize = 0;
    totalFeatSize = 0;

    int idx = 0;

    FeatureType *wordFeature = new WordFeature();
    wordFeature->getDictionaries(goldSet);
    dicSize = wordFeature->size();
    featureSize = 5;
    embeddingSize = 50;
    FeatureEmbedding *wordFeatEmb = new FeatureEmbedding(dicSize, featureSize, embeddingSize, initRange);
    featSizeOfFeatType.push_back(featureSize);
    featTypes.push_back(wordFeature);
    featEmbs.push_back(wordFeatEmb);
    WORDFEATIDX = idx++;
    coreFeatNum += featureSize;
    totalFeatSize += featureSize * embeddingSize;

    FeatureType *capFeature = new CapitalFeature();
    capFeature->getDictionaries(goldSet);
    dicSize = capFeature->size();
    featureSize = 1;
    embeddingSize = 5;
    FeatureEmbedding *capFeatEmb = new FeatureEmbedding(dicSize, featureSize, embeddingSize, initRange);
    featSizeOfFeatType.push_back(featureSize);
    featTypes.push_back(capFeature);
    featEmbs.push_back(capFeatEmb);
    CAPFEATIDX = idx++;
    coreFeatNum += featureSize;
    totalFeatSize += featureSize * embeddingSize;

    featTypeNum = idx;
}

void FeatureManager::extractFeature(State &state, Instance &inst, FeatureVector &features) {
    auto getWordIndex = [&state, &inst, this](int index) -> int {
        if (index < 0 || index >= state.m_nLen) {
            return this->featTypes[WORDFEATIDX]->nullIdx;
        }

        return inst.wordCache[index];
    };

    // auto getTagIndex = [&state, &inst, this](int index) -> int {
    //     if (index < 0 || index >= state.m_nLen) {
    //         return labelFeature->nullIdx;
    //     }

    //     return inst.tagCache[index];
    // };

    auto getCapfeatIndex = [&state, &inst, this](int index) -> int {
        if (index < 0 || index >= state.m_nLen) {
            return this->featTypes[CAPFEATIDX]->nullIdx;
        }

        return inst.capfeatCache[index];
    };

    int currentIndex = state.m_nIndex + 1;
    int IDIdx = 0;

    features.resize(featTypeNum);

    auto &wordFeature = features[WORDFEATIDX];
    wordFeature.resize(featSizeOfFeatType[WORDFEATIDX]);
    int neg2UniWord   = getWordIndex(currentIndex - 2);
    int neg1UniWord   = getWordIndex(currentIndex - 1);
    int pos0UniWord   = getWordIndex(currentIndex);
    int pos1UniWord   = getWordIndex(currentIndex + 1);
    int pos2UniWord   = getWordIndex(currentIndex + 2);
    IDIdx = 0; 
    wordFeature[IDIdx++] = neg2UniWord;
    wordFeature[IDIdx++] = neg1UniWord;
    wordFeature[IDIdx++] = pos0UniWord;
    wordFeature[IDIdx++] = pos1UniWord;
    wordFeature[IDIdx++] = pos2UniWord;

    // auto &posFeature  = features[POSFEATIDX];
    // posFeature.resize(featSizeOfFeatType[POSFEATIDX]);
    // int neg2UniPos    = getTagIndex(currentIndex - 2);
    // int neg1UniPos    = getTagIndex(currentIndex - 1);
    // int pos0UniPos    = getTagIndex(currentIndex);
    // int pos1UniPos    = getTagIndex(currentIndex + 1);
    // int pos2UniPos    = getTagIndex(currentIndex + 2);
    // IDIdx = 0;
    // features[IDIdx++] = neg2UniPos;
    // features[IDIdx++] = neg1UniPos;
    // features[IDIdx++] = pos0UniPos;
    // features[IDIdx++] = pos1UniPos;
    // features[IDIdx++] = pos2UniPos;
    
    auto &capFeature  = features[CAPFEATIDX];
    capFeature.resize(featSizeOfFeatType[CAPFEATIDX]);
    int pos0UniCap    = getCapfeatIndex(currentIndex);
    //int neg2UniCap    = getCapfeatIndex(currentIndex - 2);
    //int neg1UniCap    = getCapfeatIndex(currentIndex - 1);
    //int pos1UniCap    = getCapfeatIndex(currentIndex + 1);
    //int pos2UniCap    = getCapfeatIndex(currentIndex + 2);
    IDIdx = 0; 
    capFeature[IDIdx++] = pos0UniCap;
    // features[IDIdx++] = neg2UniCap;
    // features[IDIdx++] = neg1UniCap;
    // features[IDIdx++] = pos1UniCap;
    // features[IDIdx++] = pos2UniCap;
}

void FeatureManager::generateInstanceCache(Instance &inst) {
    inst.wordCache.resize(inst.input.size());
    inst.tagCache.resize(inst.input.size());
    inst.capfeatCache.resize(inst.input.size());

    int index = 0;
    for (auto &e : inst.input) {
        int wordIdx = featTypes[WORDFEATIDX]->feat2FeatIdx(WordFeature::processWord(e.first));
        int tagIdx = posFeature->feat2FeatIdx(e.second);
        int capfeatIdx = featTypes[CAPFEATIDX]->feat2FeatIdx(e.first);

        inst.wordCache[index] = wordIdx;
        inst.tagCache[index] = tagIdx;
        inst.capfeatCache[index] = capfeatIdx;

        index++;
    }
}

int FeatureManager::readPretrainEmbeddings(std::string &pretrainFile) {
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

    for (auto &wordPair : featTypes[WORDFEATIDX]->m_mFeat2Idx) {
        auto ret = pretrainWords.find(wordPair.first);

        if (pretrainWords.end() != ret) {
            featEmbs[WORDFEATIDX]->getPreTrain(wordPair.second, pretrainEmbs[ret->second]);
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
            int labelIdx = labelFeature->feat2FeatIdx(w.label);

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
            FeatureVector features(featTypes, featEmbs);
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
        for (int featTypeIndex = 0; featTypeIndex < featVector.size(); featTypeIndex++) {
            FeatureType &curFeatType     = *(featVector.featTypes[featTypeIndex]);
            FeatureEmbedding &curFeatEmb = *(featVector.featEmbs[featTypeIndex]);
            std::vector<int> &curFeatVec = featVector.features[featTypeIndex];

            for (int featureIndex = 0; featureIndex < curFeatVec.size(); featureIndex++) {
                for (int embIndex = 0; embIndex < curFeatEmb.embeddingSize; embIndex++) {
                    input[beamIndex][inputIndex++] = curFeatEmb[curFeatVec[featureIndex]][embIndex];
                }
            }
        }
		// for(unsigned featureIndex = 0; featureIndex < featVecs[ beamIndex ].size(); featureIndex++){ // for every feature
        //     for(unsigned embIndex = 0; embIndex < embeddingSize; embIndex++) // for every doubel in a feature embedding{}
        //     {
        //         if( featVecs[beamIndex][featureIndex] >= featEmbeddings.size() ){

        //             std::cout<<"out of mem!"<<featVecs[beamIndex][featureIndex]<<" "<<featEmbeddings.size()<<beamIndex <<" "<<featureIndex<<std::endl;
        //         }
        //         input[beamIndex][inputIndex++] = featEmbeddings[ featVecs[beamIndex][featureIndex] ][ embIndex ];
        //     }
	// }
    }
}
