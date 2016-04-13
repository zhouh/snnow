//
// Created by zhouh on 16-3-29.
//

#ifndef SNNOW_STATE_H
#define SNNOW_STATE_H

#include "Action.h"

/**
 *   the state object for the transition-based natural langurage processing
 */
class State{

    // the index of the current state in the beam
    int index_in_beam = 0;

    //! score of stack
    double score;

    //! Previous state of the current state
    State * previous;

    //! the last action to generate the state
    Action last_action;

    //! the state is gold?
    bool be_gold;
};

#endif //MYPROJECT_STATE_H
