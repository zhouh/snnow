/*************************************************************************
	> File Name: DictManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:26:31 PM CST
 ************************************************************************/
#include "DictManager.h"
// DictManager
const std::string DictManager::nullstr = "-NULL-";
const std::string DictManager::unknownstr = "-UNKNOWN-";

//WordDictManager
const std::string WordDictManager::numberstr = "-NUMBER-";
void WordDictManager::makeDictionaries(const ChunkedDataSet &goldSet) {
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

    numberIdx = idx; m_mElement2Idx[numberstr] = idx++; m_lKnownElements.push_back(numberstr);
    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
    for (auto &w : wordSet) {
        m_mElement2Idx[w] = idx++; m_lKnownElements.push_back(w);
    }
}

int WordDictManager::element2Idx(const std::string &s) {
    if (isNumber(s)) {
        return numberIdx;
    }

    auto it = m_mElement2Idx.find(s);

    return (it == m_mElement2Idx.end()) ? unkIdx : it->second;
}

std::string WordDictManager::processWord(const std::string &word) {
    std::string low_word(word);

    std::transform(low_word.begin(), low_word.end(), low_word.begin(), ::tolower);

    return low_word;
}

bool WordDictManager::isNumber(const std::string &word) {
    for (char ch : word){
        if (!isdigit(ch)) {
            return false;
        }
    }

    return true;
}

// POSDictManager
void POSDictManager::makeDictionaries(const ChunkedDataSet &goldSet) {
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

    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
    for (auto &t : posSet) {
        m_mElement2Idx[t] = idx++; m_lKnownElements.push_back(t);
    }
}

// LabelDictManager
int LabelDictManager::element2Idx(const std::string &s) {
    auto it = m_mElement2Idx.find(s);

    if (it == m_mElement2Idx.end()) {
        std::cerr << "Chunk label not found: " << s << std::endl;
        exit(0);
    }

    return it->second;
}

void LabelDictManager::makeDictionaries(const ChunkedDataSet &goldSet) {
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
        m_mElement2Idx[l] = idx++, m_lKnownElements.push_back(l);
    }
    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
}

// CapitalDictManager
const std::string CapitalDictManager::noncapitalstr = "-NONCAPITAL-";
const std::string CapitalDictManager::allcapitalstr = "-ALLCAPITAL-";
const std::string CapitalDictManager::firstlettercapstr = "-FIRSTLETTERCAPITAL-";
const std::string CapitalDictManager::hadonecapstr = "-HADONECAPITAL-";
int CapitalDictManager::element2Idx(const std::string &s) {
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
        return m_mElement2Idx[noncapitalstr];
    }

    if (isAllcap) {
        return m_mElement2Idx[allcapitalstr];
    }

    if (isFirstCap) {
        return m_mElement2Idx[firstlettercapstr];
    }

    if (isHadCap) {
        return m_mElement2Idx[hadonecapstr];
    }

    std::cerr << "word2CapfeatIdx wrong: " << s << std::endl;
    exit(1);

}

void CapitalDictManager::makeDictionaries(const ChunkedDataSet &goldSet) {
    int idx = 0;

    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
    m_mElement2Idx[noncapitalstr] = idx++; m_lKnownElements.push_back(noncapitalstr);
    m_mElement2Idx[allcapitalstr] = idx++; m_lKnownElements.push_back(allcapitalstr);
    m_mElement2Idx[firstlettercapstr] = idx++; m_lKnownElements.push_back(firstlettercapstr);
    m_mElement2Idx[hadonecapstr] = idx++; m_lKnownElements.push_back(hadonecapstr);
#ifdef DEBUG
    std::cerr << "  labelSet size: " << m_lKnownElements.size() << std::endl;
#endif
}
