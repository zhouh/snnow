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
#include <sstream>
#include <cctype>
#include <algorithm>
#include <memory>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

class Dictionary;

typedef std::vector< std::shared_ptr< Dictionary > > DictionaryVectorPtrs;

/**
 * The dictionary object to matain the mapping from the
 * sparse atomic feature the index in the feature embedding matrix
 *
 */
class Dictionary {
public:
    std::vector<std::string> known_strings;
    std::tr1::unordered_map<std::string, int> str_2_index_map;

    int unk_index;
    int null_index;

    const int c_dictionary_begin_index = 0;
    const std::string c_unk_str = "_Dict_UNK_Str_";

public:
    std::string dictionary_name;

public:
    Dictionary()  = delete;

    /**
     * get all the known strings and construct a map from them
     * @name the name of the dictionary, because we may have many dictionary in our system.
     */
    Dictionary(std::tr1::unordered_set<std::string> known_string_set, std::string name){

        this->dictionary_name = name;
        int index = 0;
        std::for_each(known_string_set.begin(), known_string_set.end(), [&] (std::string s) {
                known_strings.push_back(s);
                str_2_index_map[s] = index++;
             });

        null_index = str_2_index_map.size() ;
        unk_index = null_index + 1;
    }

    /**
     * get element index of the dictionary,
     * if not found, return the index of unk
     */
    inline int getStringIndex(std::string s) {
        auto got = str_2_index_map.find(s);
        return got == str_2_index_map.end() ? unk_index : got->second;
    }

    inline std::string getString(const int id) {
        if (id >= 0 && id < known_strings.size()) {
            return known_strings[id];
        }

        if (id == unk_index) {
            return c_unk_str;
        }

        if (id == null_index) {
            return "_NULL_str_";
        }

        std::cerr << id << " is not valid!" << std::endl;
        exit(0);
    }

    /**
     * return the null index,
     * used in the feature extractor!!
     */
    inline int getNullIndex() {
        return null_index;
    }

    inline int getUnkIndex() {
        return unk_index;
    }

    virtual ~Dictionary() {}

    int size() { // chengc modify
        return static_cast<int>(str_2_index_map.size() + 2); // + 1 for the unk string and for the null string
    }

    const std::vector<std::string>& getKnownStringVector() const {
        return known_strings;
    }

    const std::tr1::unordered_map<std::string, int>& getMap() const {
        return str_2_index_map;
    };

    virtual void saveDictionary(std::ostream & os) {
        os << "element size" << " " << known_strings.size() << std::endl;
        os << "dictionary name" << " " << dictionary_name << std::endl;

        for (std::string & e : known_strings) {
            os << e << " " << str_2_index_map[e] << std::endl;
        }
    }

    virtual void loadDictionary(std::istream &is) {
        std::string line;
        std::string tmp;
        int size;

        getline(is, line);
        std::istringstream iss(line);
        iss >> tmp >> size;
        iss >> tmp >> dictionary_name;

        null_index = size ;
        unk_index = size + 1;

        std::string element;
        int idx;
        for (int i = 0; i < size; i++) {
            getline(is, line);
            std::istringstream iss_j(line);
            iss_j >> element >> idx;

            str_2_index_map[element] = idx;
            known_strings.push_back(element);
        }
    }

    void printDict() {
        for (auto &s : known_strings) {
            std::cerr << "\t" <<  s << ": " << str_2_index_map[s] << std::endl;
        }
    }

};


#endif
