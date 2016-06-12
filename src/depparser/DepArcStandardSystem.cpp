//
// Created by zhouh on 16-4-1.
//

#include "DepArcStandardSystem.h"

std::string DepArcStandardSystem::c_root_str = "_root_str_";

// the move action is a simple call to do action according to the action code
void DepArcStandardSystem::Move(State & state, Action& action) {

    state = static_cast<DepParseState>(state);

    switch (action.getActionType()){
        case (DepParseAction::shift_type) : {
            Shift(state);
            break;
        }
        case(DepParseAction::left_type) : {
            ArcLeft(state, action);
            break;
        }
        case(DepParseAction::right_type) {
            ArcRight(state, action);
            break;
        }
    }

}

//-----------------------------------------------------------------------------
void DepArcStandardSystem::getValidActs(State & state, std::vector<int> & retval) {

    state = static_cast<DepParseState>(state);

    retval.resize(DepParseShiftReduceActionFactory::total_action_num, 0);
    retval[action_factory_ptr->makeAction(DepParseAction::left_type, rootLabelIndex)->getActionCode()] = -1; //left-root is unvalid

    int stack_size = state.m_Stack.size();
    int queue_size = state.len_ - state.m_nNextWord;

    //shift
    if (queue_size <= 0)
        retval[DepParseShiftReduceActionFactory::shift_action.getActionCode()] = -1;

    int stack_left = state.stack2top();
    //reduce
    if (stack_size >= 2 && state.stack2top()!= 0) { // all reduce is valid except reduce root
        retval[action_factory_ptr->makeAction(DepParseAction::right_type, rootLabelIndex)->getActionCode()] = -1;
        retval[action_factory_ptr->makeAction(DepParseAction::left_type, rootLabelIndex)->getActionCode()] = -1;
        return;
    }
    else {
        retval.resize(DepParseShiftReduceActionFactory::total_action_num, -1);

        if (queue_size > 0)
            retval[DepParseShiftReduceActionFactory::shift_action.getActionCode()] = 0;

        if( state.stack2top() == 0 && queue_size == 0) //except right reduce root
            retval[action_factory_ptr->makeAction(DepParseAction::right_type, rootLabelIndex)->getActionCode()] = 0;
        return;
    }

    return;
}

Action* DepArcStandardSystem::StandardMove(State & state, const Output& tree) {

    state = static_cast<DepParseState>(state);
    tree = static_cast<DepParseTree>(tree);

    if (state.complete()) {
        std::cerr << "The parsing state is completed!" << std::endl;
        exit(1);
    }

    int w2 = state.stacktop();
    int w1 = state.stack2top();
    int stackSize = state.stacksize();

    if( stackSize >= 2 && tree.nodes[w1].head == w2)
        return DepParseShiftReduceActionFactory::makeAction(
                DepParseAction::left_type,
                dep_label_map_ptr[tree.nodes[w1].label]
        );
    if( stackSize >= 2 && tree.nodes[w2].head == w1 && !state.hasChildOnQueue(w2, tree) )
        return DepParseShiftReduceActionFactory::makeAction(
                DepParseAction::right_type,
                dep_label_map_ptr[tree.nodes[w2].label]
        );
    return DepParseShiftReduceActionFactory::shift_action;
}

void DepArcStandardSystem::StandardMoveStep(State & state, const Output& tree) {
    auto action = StandardMove(state, tree);
    Move(state, action);
}

// we want to pop the root item after the whole tree done
// on the one hand this seems more natural
// on the other it is easier to score
void DepArcStandardSystem::GenerateOutput(const State& state, const Input& input, Output& output) {

    state = static_cast<DepParseState>(state);
    output = static_cast<DepParseTree>(output);

    std::cout<<"generate tree"<<std::endl;
    for (int i = 1; i < state.len_; ++i) {
        output.setHead(i, state.m_lHeads[i]);
        output.setLabel(i, known_labels[state.m_lLabels[i]]);
    }

    std::cout<<"generate tree end"<<std::endl;
}

/*
 * Perform Arc-Left operation in the arc-standard algorithm
 */
void DepArcStandardSystem::ArcLeft(DepParseState & state, DepParseAction& action) {
    // At least, there must be two elements in the stack.
    assert(state.stacksize() > 1);

    int stack_size = state.stacksize();
    int top0 = state.stacktop();
    int top1 = state.stack2top();

    state.popStack();
    state.setStackTop(top0);

    state.m_lHeads[top1] = top0;
    state.m_lLabels[top1] = action.getActionLabel();

    if (state.m_lDepsL[top0] == empty_arc) {
        state.m_lDepsL[top0] = top1;
    } else if (top1 < state.m_lDepsL[top0]) {
        state.m_lDepsL2[top0] = state.m_lDepsL[top0];
        state.m_lDepsL[top0] = top1;
    } else if (top1 < state.m_lDepsL2[top0]) {
        state.m_lDepsL2[top0] = top1;
    }

    state.last_action = action;
}

/*
 * Perform the arc-right operation in arc-standard
 */
void DepArcStandardSystem::ArcRight(DepParseState & state, DepParseAction& action) {

    assert(state.stacksize() > 1);


    int stack_size = state.stacksize();
    int top0 = state.stacktop();
    int top1 = state.stack2top();

    state.popStack();
    state.m_lHeads[top0] = top1;
    state.m_lLabels[top0] = action.getActionLabel();

    if (state.m_lDepsR[top1] == empty_arc) {
        state.m_lDepsR[top1] = top0;
    } else if (state.m_lDepsR[top1] < top0) {
        state.m_lDepsR2[top1] = state.m_lDepsR[top1];
        state.m_lDepsR[top1] = top0;
    } else if (state.m_lDepsR2[top1] < top0) {
        state.m_lDepsR2[top1] = top0;
    }

    state.last_action = action;
}

/* 
 * the shift action does pushing
 */
void DepArcStandardSystem::Shift(DepParseState& state) {
    state.m_Stack.push_back(state.m_nNextWord);
    state.m_nNextWord++;
    //state.ClearNext();
    state.last_action = action_factory_ptr->shift_action;
}