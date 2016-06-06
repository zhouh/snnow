//
// Created by zhouh on 16-3-29.
//

#ifndef SNNOW_TRANSITIONSYSTEM_H
#define SNNOW_TRANSITIONSYSTEM_H

#include "Action.h"
#include "Input.h"
#include "Output.h"
#include "State.h"

class TransitionSystem{

public:

    TransitionSystem(){
    }

    virtual void Move(State &state, const Action& action) = 0;

    /**
     * return the vector of whether the action is unvalid
     */
    virtual void getValidActs(State& state, std::vector<int>& ret_val) = 0;

    virtual Action* StandardMove(State& state, const Output& tree);

    virtual void GenerateOutput(const State& state, const Input& input, Output& output);

    virtual void StandardMoveStep(State& state, const Output& tree);
};
#endif //MYPROJECT_TRANSITIONSYSTEM_H
