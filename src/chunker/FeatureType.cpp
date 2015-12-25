/*************************************************************************
	> File Name: FeatureType.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:26:31 PM CST
 ************************************************************************/
#include "FeatureType.h"
// FeatureType
const std::string FeatureType::nullstr = "-NULL-";
const std::string FeatureType::unknownstr = "-UNKNOWN-";

//WordFeature
const std::string WordFeature::numberstr = "-NUMBER-";
void WordFeature::getDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> wordSet;

    for (auto &sent : goldSet) {
        for (auto &cw : sent.getChunkedWords()) {
            wordSet.insert(processWord(cw.word));
        }
    }
#ifdef DEBUG
    std::cerr << "  wordSet size: " << wordSet.size() << std::endl;
#endif
    int idx = 0;

    numberIdx = idx; m_mFeat2Idx[numberstr] = idx++; m_lKnownFeatures.push_back(numberstr);
    nullIdx = idx; m_mFeat2Idx[nullstr] = idx++; m_lKnownFeatures.push_back(nullstr);
    unkIdx = idx; m_mFeat2Idx[unknownstr] = idx++; m_lKnownFeatures.push_back(unknownstr);
    for (auto &w : wordSet) {
        m_mFeat2Idx[w] = idx++; m_lKnownFeatures.push_back(w);
    }
}

int WordFeature::feat2FeatIdx(const std::string &s) {
    if (isNumber(s)) {
        return numberIdx;
    }

    auto it = m_mFeat2Idx.find(s);

    return (it == m_mFeat2Idx.end()) ? unkIdx : it->second;
}

std::string WordFeature::processWord(const std::string &word) {
    std::string low_word(word);

    std::transform(low_word.begin(), low_word.end(), low_word.begin(), ::tolower);

    return low_word;
}

bool WordFeature::isNumber(const std::string &word) {
    for (char ch : word){
        if (!isdigit(ch)) {
            return false;
        }
    }

    return true;
}

// POSFeature
void POSFeature::getDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> posSet;

    for (auto &sent : goldSet) {
        for (auto &cw : sent.getChunkedWords()) {
            posSet.insert(cw.tag);
        }
    }
#ifdef DEBUG
    std::cerr << "  tagSet size: " << posSet.size() << std::endl;
#endif
    int idx = 0;

    nullIdx = idx; m_mFeat2Idx[nullstr] = idx++; m_lKnownFeatures.push_back(nullstr);
    unkIdx = idx; m_mFeat2Idx[unknownstr] = idx++; m_lKnownFeatures.push_back(unknownstr);
    for (auto &t : posSet) {
        m_mFeat2Idx[t] = idx++; m_lKnownFeatures.push_back(t);
    }
}

// LabelFeature
int LabelFeature::feat2FeatIdx(const std::string &s) {
    auto it = m_mFeat2Idx.find(s);

    if (it == m_mFeat2Idx.end()) {
        std::cerr << "Chunk label not found: " << s << std::endl;
        exit(0);
    }

    return it->second;
}

void LabelFeature::getDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> labelSet;

    for (auto &sent: goldSet) {
        for (auto &cw : sent.getChunkedWords()) {
            labelSet.insert(cw.label);
        }
    }
#ifdef DEBUG
    std::cerr << "  labelSet size: " << labelSet.size() << std::endl;
#endif
    int idx = 0;

    for (auto &l : labelSet) {
        m_mFeat2Idx[l] = idx++, m_lKnownFeatures.push_back(l);
    }
    nullIdx = idx; m_mFeat2Idx[nullstr] = idx++; m_lKnownFeatures.push_back(nullstr);
    unkIdx = idx; m_mFeat2Idx[unknownstr] = idx++; m_lKnownFeatures.push_back(unknownstr);
}

// CapitalFeature
const std::string CapitalFeature::noncapitalstr = "-NONCAPITAL-";
const std::string CapitalFeature::allcapitalstr = "-ALLCAPITAL-";
const std::string CapitalFeature::firstlettercapstr = "-FIRSTLETTERCAPITAL-";
const std::string CapitalFeature::hadonecapstr = "-HADONECAPITAL-";
int CapitalFeature::feat2FeatIdx(const std::string &s) {
    bool isNoncap = true;
    bool isAllcap = true;
    bool isFirstCap = false;
    bool isHadCap  = false;

    if (isupper(s[0])) {
        isFirstCap = true;
    }

    for (char ch : s) {
        if (isupper(ch)) {
            isHadCap = true;
            isNoncap = false;
        } else {
            isAllcap = false;
        }
    }

    if (isNoncap) {
        return m_mFeat2Idx[noncapitalstr];
    }

    if (isAllcap) {
        return m_mFeat2Idx[allcapitalstr];
    }

    if (isFirstCap) {
        return m_mFeat2Idx[firstlettercapstr];
    }

    if (isHadCap) {
        return m_mFeat2Idx[hadonecapstr];
    }

    std::cerr << "word2CapfeatIdx wrong: " << s << std::endl;
    exit(1);

}

void CapitalFeature::getDictionaries(const ChunkedDataSet &goldSet) {
    int idx = 0;

    nullIdx = idx; m_mFeat2Idx[nullstr] = idx++; m_lKnownFeatures.push_back(nullstr);
    unkIdx = idx; m_mFeat2Idx[unknownstr] = idx++; m_lKnownFeatures.push_back(unknownstr);
    m_mFeat2Idx[noncapitalstr] = idx++; m_lKnownFeatures.push_back(noncapitalstr);
    m_mFeat2Idx[allcapitalstr] = idx++; m_lKnownFeatures.push_back(allcapitalstr);
    m_mFeat2Idx[firstlettercapstr] = idx++; m_lKnownFeatures.push_back(firstlettercapstr);
    m_mFeat2Idx[hadonecapstr] = idx++; m_lKnownFeatures.push_back(hadonecapstr);
#ifdef DEBUG
    std::cerr << "  labelSet size: " << m_lKnownFeatures.size() << std::endl;
#endif
}

