//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_DEPPARSEMACRO_H
#define SNNOW_DEPPARSEMACRO_H

#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#define XPU cpu;
#define DEBUG

#define EMPTY_ARC -1;
#define EMPTY_LABEL -1;

typedef real_t float;
typedef std::vector<std::string> StringArray;
typedef std::tr1::unordered_map<std::string, int> String2IndexMap;
typedef std::tr1::unordered_set<std::string> StringSet;

const std::string c_unknow_string = "-UNKNOW-";
const std::string c_null_string = "-NULL-";
const std::string c_root_string = "-ROOT-";


#endif //SNNOW_DEPPARSEMACRO_H
