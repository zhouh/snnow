//
// Created by zhouh on 16-3-28.
//

#ifndef SNNOW_SCOREDSTATE_H
#define SNNOW_SCOREDSTATE_H

/**
 *
 */
struct ScoredTransition {
    void operator()(State* s, int a, double sc, bool g){
        state = s;
        next_action = a;
        score = sc;
        transition_is_gold = g;

    }
    //! The pointer to the source state;
    State* state;
    //! The compile action applied to the source state;
    int next_action;
    //! The resulted in score.
    double score;
    // the combined transition and action is gold ?
    bool transition_is_gold;
};

#endif //SNNOW_SCOREDSTATE_H
