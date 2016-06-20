/*************************************************************************
	> File Name: SeqLabelerStandardSystem.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:36:17 PM CST
 ************************************************************************/
#ifndef SNNOW_SEQLABELERSTANDARDSYSTEM_H
#define SNNOW_SEQLABELERSTANDARDSYSTEM_H

#include "base/TransitionSystem.h"

class SeqLabelerStandardSystem : public TransitionSystem {
public:
    void Move(State& state, Action& action) {

    }

    /**
     * return the vector of whether the action is unvalid
     */
    void getValidActs(State& state, std::vector<int>& ret_val){

    }

    Action* StandardMove(State& state, Output& tree) {

    }

    void GenerateOutput(State& state, Input& input, Output& output) {

    }

    void StandardMoveStep(State& state, Output& tree) {

    }
};

#endif // SNNOW_SEQLABELERSTANDARDSYSTEM_H
