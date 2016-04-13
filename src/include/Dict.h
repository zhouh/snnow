/*************************************************************************
	> File Name: Dictionary.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 24 Dec 2015 03:24:57 PM CST
 ************************************************************************/
#ifndef _SNNOW_COMMON_DICTIONARY_H_
#define _SNNOW_COMMON_DICTIONARY_H_

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

class Dictionary {
protected:
    std::vector<std::string> m_lKnownElements;
    std::tr1::unordered_map<std::string, int> m_mElement2Idx;
    std::string dictionary_name;

public:
    const static int c_null_index = 0;
    const static int c_unk_index = 1;
    const static int c_dictionary_begin_index = 2;

    static const std::string c_null_str = "_Dict_NULL_Str_";
    static const std::string c_unk_str = "_Dict_UNK_Str_";

public:
    Dictionary() {}

    Dictionary(std::vector<std::string> element_list, std::string name){
        this->dictionary_name = name;
        m_lKnownElements = element_list;
        int index = c_dictionary_begin_index;
        for(unsigned i = 0; i < element_list.size(); ++i)
            m_mElement2Idx[element_list[i]] = index++;
    }

    virtual ~Dictionary() {}

    int size() {
        return static_cast<int>(m_mElement2Idx.size());
    }

    const std::vector<std::string>& getKnownElements() const {
        return m_lKnownElements;
    }

    const std::tr1::unordered_map<std::string, int>& getWord2IdxMap() const {
        return m_mElement2Idx;
    }

    virtual int element2Idx(const std::string &s) const {
        auto it = m_mElement2Idx.find(s);

        return (it == m_mElement2Idx.end()) ? c_unk_index: it->second;
    }

    virtual void makeDictionaries(const ChunkedDataSet &goldSet) = 0;

    virtual void saveDictionary(std::ostream &os) {
        os << "elementSize" << " " << m_lKnownElements.size() << std::endl;

        for (std::string &e : m_lKnownElements) {
            os << e << " " << m_mElement2Idx[e] << std::endl;
        }
    }

    virtual void loadDictionary(std::istream &is) {
        std::string line;
        std::string tmp;
        int size;
    
        getline(is, line);
        std::istringstream iss(line);
        iss >> tmp >> size;
    
        std::string element;
        int idx;
        for (int i = 0; i < size; i++) {
            getline(is, line);
            std::istringstream iss_j(line);
            iss_j >> element >> idx;

            processElementAndIdx(element, idx);
    
            m_mElement2Idx[element] = idx;
            m_lKnownElements.push_back(element);
        }
    }

    void printDict() {
        for (auto &s : m_lKnownElements) {
            std::cerr << "  " <<  s << ": " << m_mElement2Idx[s] << std::endl;
        }
    }

private:
    Dictionary(const Dictionary &dManager) = delete;
    Dictionary& operator= (const Dictionary &dManager) = delete;

};


#endif
