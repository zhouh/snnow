/*************************************************************************
	> File Name: FeatureExtractor.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 02:55:27 PM CST
 ************************************************************************/
#include <unordered_set>
#include <memory>

#include "FeatureExtractor.h"

#define DEBUG
#ifdef DEBUG
//#define DEBUG1
#define DEBUG2
//#define DEBUG3
#define DEBUG4

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
    std::cerr << "  wordSet size: " << wordSet.size() << std::endl;
    std::cerr << "  tagSet size: " << tagSet.size() << std::endl;
    std::cerr << "  labelSet Size: " << labelSet.size() << std::endl;
#endif

    int idx = 0;

    for (auto &l : labelSet) {
        m_mLabel2Idx[l] = idx++;

#ifdef DEBUG3
        std::cout << l << ": " << idx - 1 << std::endl;
#endif
        m_lKnownLabels.push_back(l);
    }
    labelNullIdx = idx, m_mLabel2Idx[nullstr] = idx++;
#ifdef DEBUG3
        std::cout << nullstr << "[label]: " << idx - 1 << std::endl;
#endif
    labelUnkIdx = idx, m_mLabel2Idx[unknownstr] = idx++;
#ifdef DEBUG3
        std::cout << unknownstr << "[label]: " << idx - 1 << std::endl;
#endif

    wordNullIdx = idx, m_mWord2Idx[nullstr] = idx++;
#ifdef DEBUG3
        std::cout << nullstr << "[word]: " << idx - 1 << std::endl;
#endif
    wordUnkIdx = idx, m_mWord2Idx[unknownstr] = idx++;
#ifdef DEBUG3
        std::cout << unknownstr << "[word]: " << idx - 1 << std::endl;
#endif
    m_lKnownWords.push_back(nullstr);
    m_lKnownWords.push_back(unknownstr);
    for (auto &w : wordSet) {
        m_mWord2Idx[w] = idx++;
#ifdef DEBUG3
        std::cout << w << ": " << idx - 1 << std::endl;
#endif
        m_lKnownWords.push_back(w);
    }

    tagNullIdx = idx, m_mTag2Idx[nullstr] = idx++;
#ifdef DEBUG3
    std::cout << nullstr << "[tag]: " << idx - 1 << std::endl;
#endif
    tagUnkIdx = idx, m_mTag2Idx[unknownstr] = idx++;
#ifdef DEBUG3
    std::cout << unknownstr << "[tag]: " << idx - 1 << std::endl;
#endif
    m_lKnownTags.push_back(nullstr);
    m_lKnownTags.push_back(unknownstr);
    for (auto &t : tagSet) {
        m_mTag2Idx[t] = idx++;
#ifdef DEBUG3
        std::cout << t << ": " << idx - 1 << std::endl;
#endif
        m_lKnownTags.push_back(t);
    }
}

void FeatureExtractor::generateInstanceCache(Instance &inst) {
    inst.wordCache.resize(inst.input.size());
    inst.tagCache.resize(inst.input.size());

    int index = 0;
    for (auto &e : inst.input) {
        int wordIdx = word2WordIdx(e.first);
        int tagIdx = tag2TagIdx(e.second);

        inst.wordCache[index] = wordIdx;
        inst.tagCache[index] = tagIdx;

        index++;
    }
}

void FeatureExtractor::generateInstanceSetCache(InstanceSet &instSet) {
    for (auto &inst : instSet) {
        generateInstanceCache(inst);
    }
}

void FeatureExtractor::extractFeature(State &state, Instance &inst, std::vector<int> &features) {
    auto getWordIndex = [&state, &inst, this](int index) -> int {
        if (index < 0 || index >= state.m_nLen) {
            return this->wordNullIdx;
        }

        return inst.wordCache[index];
    };

    auto getTagIndex = [&state, &inst, this](int index) -> int {
        if (index < 0 || index >= state.m_nLen) {
            return this->tagNullIdx;
        }

        return inst.tagCache[index];
    };

    int IDIdx = 0;
    int currentIndex = state.m_nIndex + 1;

    int neg3UniWord   = getWordIndex(currentIndex - 3);
    int neg2UniWord   = getWordIndex(currentIndex - 2);
    int neg1UniWord   = getWordIndex(currentIndex - 1);
    int pos1UniWord   = getWordIndex(currentIndex);
    int pos2UniWord   = getWordIndex(currentIndex + 1);
    int pos3UniWord   = getWordIndex(currentIndex + 2);

    features[IDIdx++] = neg3UniWord;
    features[IDIdx++] = neg2UniWord;
    features[IDIdx++] = neg1UniWord;
    features[IDIdx++] = pos1UniWord;
    features[IDIdx++] = pos2UniWord;
    features[IDIdx++] = pos3UniWord;

    int neg3UniPos    = getTagIndex(currentIndex - 3);
    int neg2UniPos    = getTagIndex(currentIndex - 2);
    int neg1UniPos    = getTagIndex(currentIndex - 1);
    int pos1UniPos    = getTagIndex(currentIndex);
    int pos2UniPos    = getTagIndex(currentIndex + 1);
    int pos3UniPos    = getTagIndex(currentIndex + 2);

    features[IDIdx++] = neg3UniPos;
    features[IDIdx++] = neg2UniPos;
    features[IDIdx++] = neg1UniPos;
    features[IDIdx++] = pos1UniPos;
    features[IDIdx++] = pos2UniPos;
    features[IDIdx++] = pos3UniPos;
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
            int labelIdx = label2LabelIdx(w.label);
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

#ifdef DEBUGX
            std::cout << "j = " << j << std::endl;
            for (int fi = 0; fi < features.size(); fi++) {
                std::cout << features[fi] << " ";
            }
            std::cout << std::endl;
            char tch;
            std::cin >> tch;
#endif
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
#ifdef DEBUGX
    std::cout << "pretrain file path is: " << pretrainFile << std::endl;
#endif
    std::string line;
    std::ifstream in(pretrainFile);
    getline(in, line); //TODO dirrent from zhouh

    int index = 0;
    while (getline(in, line)) {
#ifdef DEBUGX
    std::cout << "line: " << line << std::endl;
#endif
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
