//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DEPPARSEINPUT_H
#define SNNOW_DEPPARSEINPUT_H

#include <vector>
#include <algorithm>
#include <string>


#include "base/Input.h"

/**
 * The dep parse input do not includes the ROOT nodes.
 */
class DepParseInput : public Input, public std::vector<std::pair<std::string, std::string>>{

public:
    /*
     * cache for fast obtaining the index of tag and word
     */
    std::vector<int> tag_cache;
    std::vector<int> word_cache;

};


inline std::ostream & operator << (std::ostream &os, const DepParseInput &input) {

    // output from node 1, skip root node
    for(unsigned i = 0; i < input.size(); i++)
        os <<input[i].first <<"("<<  input[i].second<<")\t";
    os << std::endl;

    return os ;
}



#endif //SNNOW_DEPPARSEINPUT_H
