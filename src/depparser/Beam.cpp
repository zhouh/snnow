/*************************************************************************
	> File Name: src/depparser/Beam.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 08/09/15 15:09:31
 ************************************************************************/

#include "State.h"
#include "DepTree.h"
/** 
 * std::heap is a max heap, and the compare function return first element is less that the second.
 * But here, we need a min heap, so return x.score > y.score
 */
bool
ScoredTransitionMore(const CScoredTransition& x, const CScoredTransition& y) {
  return x.score > y.score;
}

