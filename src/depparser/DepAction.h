/*
 * DepAction.h
 *
 *  Created on: Jul 3, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_DEPACTION_H_
#define DEPPARSER_DEPACTION_H_

#include "Config.h"
#include "assert.h"

// SH [AL+LABEL] [AR+LABEL]
enum StackActions {
  kShift = 0,
  kArcLeftFirst = 1,
  kArcRightFirst = CConfig::nLabelNum + 2,
  kActNum = CConfig::nLabelNum * 2 + 1,
};

const int empty_arc = -1;
const int empty_label = -1;

static std::string unknow = "-UNKNOW-";
static std::string null = "-NULL-";
static std::string root = "-ROOT-";

static int rootLabelIndex = 1;

/**
 *   return the action code
 */
static unsigned
EncodeAction(const StackActions & action,
             const int & label = 0) {

  if (action == kShift)
	  return action;
  else
	  return action + label;
}

/**
 *   get the action type
 */
static unsigned
DecodeUnlabeledAction(const unsigned & action) {
  assert(action < kActNum);

  if (action < kArcLeftFirst)
    return kShift;
  else if (action < kArcRightFirst)
    return kArcLeftFirst;
  else
    return kArcRightFirst;

}

/**
 *  get the dependency label ID
 */
static unsigned
DecodeLabel(const unsigned & action) {
  assert(action < kActNum);
  return action - DecodeUnlabeledAction(action);
}

#endif /* DEPPARSER_DEPACTION_H_ */
