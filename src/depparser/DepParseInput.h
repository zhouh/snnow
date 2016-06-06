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


#endif //SNNOW_DEPPARSEINPUT_H
