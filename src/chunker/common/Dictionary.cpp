/*************************************************************************
	> File Name: Dictionary.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Fri 25 Dec 2015 03:26:31 PM CST
 ************************************************************************/
#include "Dictionary.h"

// Dictionary
const std::string Dictionary::nullstr = "-NULL-";
const std::string Dictionary::unknownstr = "-UNKNOWN-";

//WordDictionary
const std::string WordDictionary::numberstr = "NUMBER";
void WordDictionary::makeDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> wordSet;

    for (auto &sent : goldSet) {
        for (auto &cw : sent.getLabeledTerms()) {
            wordSet.insert(processWord(cw.word));
        }
    }
    int idx = 0;

    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
    for (auto &w : wordSet) {
        m_mElement2Idx[w] = idx++; m_lKnownElements.push_back(w);
    }
}

int WordDictionary::element2Idx(const std::string &s) const {
    auto it = m_mElement2Idx.find(s);

    return (it == m_mElement2Idx.end()) ? unkIdx : it->second;
}

std::string WordDictionary::processWord(const std::string &word) {
    std::string low_word(word);

    std::transform(word.begin(), word.end(), low_word.begin(), ::tolower);

    low_word = replaceNumber(low_word);

    return low_word;
}

std::string WordDictionary::replaceNumber(const std::string &word) {
    std::string ret;

    bool isNumber = false;
    for (char ch : word) {
        if (isdigit(ch)) {
            isNumber = true;
        } else {
            if (isNumber) {
                ret += numberstr;
            }

            isNumber = false;
            ret.push_back(ch);
        }
    }
    if (isNumber) {
        ret += nullstr;
    }

    return ret;
}

// POSDictionary
void POSDictionary::makeDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> posSet;

    for (auto &sent : goldSet) {
        for (auto &cw : sent.getLabeledTerms()) {
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

// LabelDictionary
int LabelDictionary::element2Idx(const std::string &s) const {
    auto it = m_mElement2Idx.find(s);

    if (it == m_mElement2Idx.end()) {
        std::cerr << "Chunk label not found: " << s << std::endl;
        exit(0);
    }

    return it->second;
}

void LabelDictionary::makeDictionaries(const ChunkedDataSet &goldSet) {
    using std::unordered_set;
    using std::string;

    unordered_set<string> labelSet;

    for (auto &sent: goldSet) {
        for (auto &cw : sent.getLabeledTerms()) {
            labelSet.insert(cw.label);
        }
    }
#ifdef DEBUG
    std::cerr << "  labelSet size: " << labelSet.size() << std::endl;
#endif
    int idx = 0;

    // The following codes's order can not be changed, because 
    // the ID is binded to ActionSystem's actionID system.
    for (auto &l : labelSet) {
        m_mElement2Idx[l] = idx++, m_lKnownElements.push_back(l);
    }
    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
}

// CapitalDictionary
const std::string CapitalDictionary::noncapitalstr = "-NONCAPITAL-";
const std::string CapitalDictionary::allcapitalstr = "-ALLCAPITAL-";
const std::string CapitalDictionary::firstlettercapstr = "-FIRSTLETTERCAPITAL-";
const std::string CapitalDictionary::hadonecapstr = "-HADONECAPITAL-";
int CapitalDictionary::element2Idx(const std::string &s) const {
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
        return m_mElement2Idx.find(noncapitalstr)->second;
    }

    if (isAllcap) {
        return m_mElement2Idx.find(allcapitalstr)->second;
    }

    if (isFirstCap) {
        return m_mElement2Idx.find(firstlettercapstr)->second;
    }

    if (isHadCap) {
        return m_mElement2Idx.find(hadonecapstr)->second;
    }

    std::cerr << "word2CapfeatIdx wrong: " << s << std::endl;
    exit(1);

}

void CapitalDictionary::makeDictionaries(const ChunkedDataSet &goldSet) {
    int idx = 0;

    nullIdx = idx; m_mElement2Idx[nullstr] = idx++; m_lKnownElements.push_back(nullstr);
    unkIdx = idx; m_mElement2Idx[unknownstr] = idx++; m_lKnownElements.push_back(unknownstr);
    m_mElement2Idx[noncapitalstr] = idx++; m_lKnownElements.push_back(noncapitalstr);
    m_mElement2Idx[allcapitalstr] = idx++; m_lKnownElements.push_back(allcapitalstr);
    m_mElement2Idx[firstlettercapstr] = idx++; m_lKnownElements.push_back(firstlettercapstr);
    m_mElement2Idx[hadonecapstr] = idx++; m_lKnownElements.push_back(hadonecapstr);
}

