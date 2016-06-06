//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_DEPPARSEMACRO_H
#define SNNOW_DEPPARSEMACRO_H

#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include "Macros.h"

#define XPU gpu;
#define DEBUG

#define EMPTY_ARC -1;
#define EMPTY_LABEL -1;

typedef std::vector<std::string> StringArray;
typedef std::tr1::unordered_map<std::string, int> String2IndexMap;
typedef std::tr1::unordered_set<std::string> StringSet;

const static int c_feature_type_num = 3;
const static int c_word_feature_dim = 50;
const static int c_tag_feature_dim = 50;
const static int c_label_feature_dim = 50;


#endif //SNNOW_DEPPARSEMACRO_H
