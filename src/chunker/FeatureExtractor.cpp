/*************************************************************************
	> File Name: FeatureExtractor.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 02:55:27 PM CST
 ************************************************************************/
#include <unordered_set>
#include <memory>

#include "FeatureExtractor.h"

//#define DEBUG
#ifdef DEBUG

//#define DEBUG1
#define DEBUG2

#endif // DEBUG

std::string FeatureExtractor::nullstr = "-NULL-";
std::string FeatureExtractor::unknownstr = "-UNKNOWN-";

void FeatureExtractor::getDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> labelSet;
    unordered_set<string> wordSet;
    unordered_set<string> tagSet;

    for (auto &sent : goldSet) {
        for (auto &cw : sent.getChunkedWords()) {
            labelSet.insert(cw.label);
            wordSet.insert(cw.word);
            tagSet.insert(cw.tag);
        }
    }

#ifdef DEBUG
    std::cerr << "wordSet size: " << wordSet.size() << std::endl;
    std::cerr << "tagSet size: " << tagSet.size() << std::endl;
    std::cerr << "labelSet Size: " << labelSet.size() << std::endl;
#endif

    int idx = 0;

    for (auto &l : labelSet) {
        m_mLabel2Idx[l] = idx++;
        m_lKnownLabels.push_back(l);
    }
    labelNullIdx = idx, m_mLabel2Idx[nullstr] = idx++;
    labelUnkIdx = idx, m_mLabel2Idx[unknownstr] = idx++;

    wordNullIdx = idx, m_mWord2Idx[nullstr] = idx++;
    wordUnkIdx = idx, m_mWord2Idx[unknownstr] = idx++;
    m_lKnownWords.push_back(nullstr);
    m_lKnownWords.push_back(unknownstr);
    for (auto &w : wordSet) {
        m_mWord2Idx[w] = idx++;
        m_lKnownWords.push_back(w);
    }

    tagNullIdx = idx, m_mTag2Idx[nullstr] = idx++;
    tagUnkIdx = idx, m_mTag2Idx[unknownstr] = idx++;
    m_lKnownTags.push_back(nullstr);
    m_lKnownTags.push_back(unknownstr);
    for (auto &t : tagSet) {
        m_mTag2Idx[t] = idx++;
        m_lKnownTags.push_back(t);
    }
}

void FeatureExtractor::generateInstanceCache(Instance &inst) {
    inst.wordCache.resize(inst.input.size());
    inst.tagCache.resize(inst.input.size());

    int index = 0;
    for (auto &e : inst.input) {
        int wordIdx = getWordIdx(e.first);
        int tagIdx = getTagIdx(e.second);

        inst.wordCache[index] = wordIdx;
        inst.tagCache[index] = tagIdx;

        index++;
    }
}

void FeatureExtractor::generateInstanceSetCache(InstanceSet &instSet) {
    for (auto &inst : instSet)
        generateInstanceCache(inst);
}

void FeatureExtractor::extractFeature(State &state, Instance &inst, std::vector<int> &features) {

}

void FeatureExtractor::generateTrainingExamples(ActionStandardSystem &transitionSystem, InstanceSet &instSet, ChunkedDataSet &goldSet, GlobalExamples &gExamples) {
    gExamples.clear();

    for (int i = 0; i < instSet.size(); i++) {
        Instance &inst = instSet[i];
        ChunkedSentence &gSent = goldSet[i];
    
        ChunkerInput &input = inst.input;
        int actNum = input.size();

        std::vector<int> labelIndexCache(gSent.size());
        int index = 0;
        for (const ChunkedWord &w : gSent.getChunkedWords()) {
            int labelIdx = getLabelIdx(w.label);
#ifdef DEBUG1
            std::cerr << w.word << " " << w.tag << " " << w.label << std::endl;
            //std::cerr << "label idx: " << labelIdx << std::endl;
#endif
            labelIndexCache[index] = labelIdx;
            index++;
        }

        std::vector<int> acts(actNum);
        std::vector<Example> examples;

        std::shared_ptr<State> state(new State());
        state->m_nLen = input.size();
        generateInstanceCache(inst);

#ifdef DEBUG1
        std::cerr << std::endl;
        std::cerr << "Sentence length: " << gSent.size() << std::endl;
        std::cerr << "Gold acts:" << std::endl;
#endif
        //generate every state of a sentence
        for (int j = 0; !state->complete(); j++) {
            std::vector<int> features(featureNum);
            std::vector<int> labels(transitionSystem.nActNum, 0);

            extractFeature(*state, inst, features);
            
            transitionSystem.generateValidActs(*state, labels);

            int goldAct = transitionSystem.standardMove(*state, gSent, labelIndexCache);
            acts[j] = goldAct;

            CScoredTransition tTrans(NULL, goldAct, 0);
            transitionSystem.move(*state, *state, tTrans);

#ifdef DEBUG1
            // std::cerr << goldAct << " : " << transitionSystem.actionIdx2LabelIdx(goldAct) << "   ";
            std::cerr << (j + 1) << ":" << goldAct << "(" << m_lKnownLabels[transitionSystem.actionIdx2LabelIdx(goldAct)] << ") -> ";
#endif
            labels[goldAct] = 1;

            Example example(features, labels);
            examples.push_back(example);
        }
#ifdef DEBUG1
        std::cerr << std::endl;
        std::cerr << std::endl;

        std::cerr << "GlobalExample: " << std::endl;
        std::cerr << "Length: " << examples.size() << std::endl;
        int t = 1;
        for (Example &e : examples) {
            for (int i = 0; i < e.labels.size(); i++) {
                if (e.labels[i] == 1) {
                    std::cerr << (t++) << ":" << i << " -> ";
                }
            }
        }
#endif
        GlobalExample tGlobalExampe(examples, acts, inst);
        gExamples.push_back(tGlobalExampe);

#ifdef DEBUG1
        char ch;
        std::cin >> ch;
#endif
    }
}

int FeatureExtractor::readPretrainEmbeddings(std::string &pretrainFile, FeatureEmbedding &fe){
    std::tr1::unordered_map<std::string, int> pretrainWords;
    std::vector<std::vector<double>> pretrainEmbs;

    std::string line;
    std::ifstream in(pretrainFile);
    getline(in, line);

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

#ifdef DEBUG2
    std::cerr << "pretrainWords's size: " << pretrainEmbs.size() << std::endl;
#endif

    for (auto &wordPair : m_mWord2Idx) {
        auto ret = pretrainWords.find(wordPair.first);

        if (pretrainWords.end() != ret) {
            fe.getPreTrain(wordPair.second, pretrainEmbs[ret->second]);
        }
    }

#ifdef DEBUG2
    std::cerr << "total words's size: " << m_mWord2Idx.size() << std::endl;
#endif

    return pretrainWords.size();
}
