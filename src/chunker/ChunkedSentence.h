/*************************************************************************
	> File Name: ChunkedSentence.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 19 Nov 2015 10:20:35 AM CST
 ************************************************************************/
#ifndef _CHUNKER_CHUNKEDSENTENCE_H_
#define _CHUNKER_CHUNKEDSENTENCE_H_

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

typedef std::vector<std::pair<std::string, std::string>> ChunkerInput;

struct ChunkedWord {
    std::string word;
    std::string tag;
    std::string label;

    ChunkedWord() {}
    ChunkedWord(std::string w, std::string t, std::string l): word(w), tag(t), label(l) {}
    ChunkedWord(std::string w, std::string t): word(w), tag(t) {}
    static std::string reviseLabel(const std::string &s) {
        std::size_t found = s.find_first_of("-");

        std::string ret = s.substr(0, found);

        if (ret == "I") {
            return ret;
        } else {
            return s;
        }
    }
};


inline std::istream& operator >> (std::istream &is, ChunkedWord &cw) {
    std::string line;

    std::getline(is, line, ' ');
    cw.word = line;

    std::getline(is, line, ' ');
    cw.tag = line;

    std::getline(is, line, ' ');
    cw.label = ChunkedWord::reviseLabel(line);

    return is;
}

inline std::ostream& operator << (std::ostream &os, const ChunkedWord &cw) {
    os << cw.word << " " << cw.tag << " " << cw.label << std::endl;

    return os;
}

class ChunkedSentence {
public:
    std::vector<ChunkedWord> m_lChunkedWords;
    int m_nLength;

    ChunkedSentence(ChunkerInput ci) {
        for (auto &e : ci) {
            ChunkedWord cw(e.first, e.second);

            m_lChunkedWords.push_back(cw);
        }

        m_nLength = m_lChunkedWords.size();
    }

    ChunkedSentence(): m_lChunkedWords(0) {}

    ~ChunkedSentence() {}

    void init(ChunkerInput &input) {
        for (auto iter = input.begin(); iter != input.end(); iter++) {
            ChunkedWord cw(iter->first, iter->second);

            m_lChunkedWords.push_back(cw);
        }

        m_nLength = m_lChunkedWords.size();
    }

    void getChunkerInput(ChunkerInput &ci) {
        ci.resize(m_nLength);

        for (int i = 0; i < m_nLength; i++) {
            const ChunkedWord &e = m_lChunkedWords[i];
            ci[i].first = e.word;
            ci[i].second = e.tag;
        }
    }

    int size() {
        return m_nLength;
    }

    const std::vector<ChunkedWord>& getChunkedWords() const {
        return m_lChunkedWords;
    }

    friend std::istream& operator >> (std::istream &is, ChunkedSentence &cs); 
    friend std::ostream& operator << (std::ostream &os, const ChunkedSentence &cs);
};

inline std::istream& operator >> (std::istream &is, ChunkedSentence &cs) {
    std::string line;

    getline(is, line);
    while (is && !line.empty()) {
        ChunkedWord cw;
        std::istringstream iss(line);
        iss >> cw;
        cs.m_lChunkedWords.push_back(cw);

        getline(is, line);
    }

    cs.m_nLength = cs.m_lChunkedWords.size();

    return is;
}

inline std::ostream& operator << (std::ostream &os, const ChunkedSentence &cs) {
    for (auto &cw : cs.m_lChunkedWords)
        os << cw;

    return os;
}

typedef std::vector<ChunkedSentence> ChunkedDataSet;
#endif

