/*************************************************************************
	> File Name: SeqLabelerMacro.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 15 Jun 2016 12:32:11 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELERMACRO_H
#define SNNOW_SEQLABELERMACRO_H

#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include "Macros.h"

#define XPU gpu;
#define DEBUG

typedef std::vector<std::string> StringArray;
typedef std::tr1::unordered_map<std::string, int> String2IndexMap;
typedef std::tr1::unordered_set<std::string> StringSet;

#endif // SNNOW_SEQLABELERMACRO_H
