/*************************************************************************
	> File Name: LabeledSequence.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 10:20:35 AM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_LABELEDSEQUENCE_H_
#define _CHUNKER_COMMON_LABELEDSEQUENCE_H_

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <assert.h>

typedef std::vector<std::pair<std::string, std::string>> SequenceInput;

// LabeledTerm
struct LabeledTerm {
    std::string word;
    std::string tag;
    std::string label;

    LabeledTerm() {}
    LabeledTerm(std::string w, std::string t, std::string l): word(w), tag(t), label(l) {}
    LabeledTerm(std::string w, std::string t): word(w), tag(t) {}
    LabeledTerm(const LabeledTerm &lt) : word(lt.word), tag(lt.tag), label(lt.label) {}
    LabeledTerm& operator= (const LabeledTerm &lt) {
        if (this == &lt) {
            return *this;
        }

        this->word = lt.word;
        this->tag = lt.tag;
        this->label = lt.label;

        return *this;
    }
};


inline std::istream& operator >> (std::istream &is, LabeledTerm &lt) {
    std::string line;

    std::getline(is, line, ' ');
    lt.word = line;

    std::getline(is, line, ' ');
    lt.tag = line;

    std::getline(is, line, ' ');
    lt.label = line;

    return is;
}

inline std::ostream& operator << (std::ostream &os, const LabeledTerm &lt) {
    os << lt.word << " " << lt.tag << " " << lt.label << std::endl;

    return os;
}

// LabeledSequence
class LabeledSequence {
private:
    std::vector<LabeledTerm> m_lLabeledTerms;
    int m_nLength;

public:

    LabeledSequence(): m_nLength(0) {}

    LabeledSequence(const SequenceInput &ci) {
        for (auto &e : ci) {
            LabeledTerm lt(e.first, e.second);

            m_lLabeledTerms.push_back(lt);
        }

        m_nLength = static_cast<int>(m_lLabeledTerms.size());
    }

    LabeledSequence(const LabeledSequence &ls): m_nLength(ls.m_nLength), m_lLabeledTerms(ls.m_lLabeledTerms) {
    }

    LabeledSequence& operator= (const LabeledSequence &ls) {
        if (this == &ls) {
            return *this;
        }
        this->m_lLabeledTerms.resize(ls.m_nLength);

        m_nLength = ls.m_nLength;

        for (int i = 0; i < m_nLength; i++) {
            m_lLabeledTerms[i] = ls.m_lLabeledTerms[i];
        }

        return *this;
    }
    ~LabeledSequence() {}

    void init(SequenceInput &input) {
        for (auto iter = input.begin(); iter != input.end(); iter++) {
            LabeledTerm lt(iter->first, iter->second);

            m_lLabeledTerms.push_back(lt);
        }

        m_nLength = m_lLabeledTerms.size();
    }

    void getSequenceInput(SequenceInput &ci) {
        ci.resize(m_nLength);

        for (int i = 0; i < m_nLength; i++) {
            const LabeledTerm &e = m_lLabeledTerms[i];
            ci[i].first = e.word;
            ci[i].second = e.tag;
        }
    }

    int size() {
        return m_nLength;
    }

    const std::vector<LabeledTerm>& getLabeledTerms() const {
        return m_lLabeledTerms;
    }

    void setLabel(const int i, const std::string &label) {
        assert (i >= 0 && i <= m_nLength);

        m_lLabeledTerms[i].label = label;
    }

    void tranform2IAttachFormat() {
        for (auto & lt : m_lLabeledTerms) {
            lt.label = toAttachLabel(lt.label);
        }
    }

    void transform2StardandBIOFormat() {
        static std::string BEGIN_I_TYPE = "ITYPE";

        std::string type = BEGIN_I_TYPE;

        for (auto & lt : m_lLabeledTerms){
            if (lt.label == "I") {
                lt.label = "I-" + type;
            } else if (lt.label[0] == 'B') {
                type = lt.label.substr(2);
            } else {
                type = BEGIN_I_TYPE;
            }
        }
    }

    friend std::istream& operator >> (std::istream &is, LabeledSequence &ls); 
    friend std::ostream& operator << (std::ostream &os, const LabeledSequence &ls);

private:
    // tranform "I-Labeltype" to "I"
    static std::string toAttachLabel(const std::string &s) {
        std::size_t found = s.find_first_of("-");

        std::string ret = s.substr(0, found);

        if (ret == "I") {
            return ret;
        } else {
            return s;
        }
    }
};


inline std::istream& operator >> (std::istream &is, LabeledSequence &ls) {
    std::string line;

    getline(is, line);
    while (is && !line.empty()) {
        LabeledTerm lt;
        std::istringstream iss(line);
        iss >> lt;
        ls.m_lLabeledTerms.push_back(lt);

        getline(is, line);
    }

    ls.m_nLength = static_cast<int>(ls.m_lLabeledTerms.size());

    ls.tranform2IAttachFormat();

    return is;
}

inline std::ostream& operator << (std::ostream &os, const LabeledSequence &ls) {
    for (auto &lt : ls.m_lLabeledTerms)
        os << lt;

    return os;
}

typedef std::vector<LabeledSequence> ChunkedDataSet;
#endif

