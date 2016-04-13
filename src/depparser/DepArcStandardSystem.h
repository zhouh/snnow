//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_DepArcStandardSystem_H
#define SNNOW_DepArcStandardSystem_H

#include <memory>

#include "DepParseMacro.h"
#include "base/TransitionSystem.h"
#include "DepParseShiftReduceAction.h"
#include "DepParseState.h"
#include "DepParseTree.h"


class DepArcStandardSystem : public TransitionSystem{

public:

    std::shared_ptr<String2IndexMap> dep_label_map_ptr;
    std::vector<std::string>& known_labels;

    int rootLabelIndex = -1;

    std::shared_ptr<DepParseShiftReduceActionFactory> action_factory_ptr;

    DepArcStandardSystem() {
        std::clog << "using the default constructor, which is not valid!" << std::endl;
    }

    DepArcStandardSystem(std::vector<std::string>& known_labels) {

        makeTransition(known_labels);
        this->known_labels = known_labels;
        action_factory_ptr.reset(new DepParseShiftReduceActionFactory( dep_label_map_ptr.operator*().size() ));
    }

    /**
     * prepare the transition system
     */
    void makeTransition(std::vector<std::string>& knowLabels){

        /*
         * construct the dict for dep label by the know label set
         */
        this->known_labels = known_labels;
        int index = 0;
        dep_label_map_ptr.reset(new std::tr1::unordered_set());
        for(int i = 0; i < knowLabels.size(); i++) {
            (*dep_label_map_ptr)[knowLabels[i]] = index++;
        }
        known_labels.insert(c_root_str);
        (*dep_label_map_ptr)[c_root_str] = index++;  // add the ROOT label, because the knowLabels does not contain it.
        action_factory_ptr.reset(new DepParseShiftReduceActionFactory( dep_label_map_ptr.operator*().size() ));


    }

    inline int getRootLabelIndex(){
        assert(rootLabelIndex != -1);
        return rootLabelIndex;
    }

    inline auto getLeftActions(){
        return action_factory_ptr->left_reduce_actions;
    }

    inline auto getRightActions(){
        return action_factory_ptr->right_reduce_actions;
    }

    inline auto getShiftAction(){
        return action_factory_ptr->shift_action;
    }

    /*
     * Perform Arc-Left operation in the arc-standard algorithm
     */
    void ArcLeft(DepParseState & state, Action& action);

    /*
     * Perform the arc-right operation in arc-standard
     */
    void ArcRight(DepParseState & state, Action& action);

    /* 
     * the shift action does pushing
     */
    void Shift(DepParseState &state);

    void Move(State & state, const Action& action);

    void getValidActs(State & state, std::vector<int>& ret_val);

    int StandardMove(State& state, const Output& tree);

    void GenerateOutput(const State& state, const Input& input, Output& output);

    void StandardMoveStep(State& state, const Output& tree);
};
#endif //SNNOW_DepArcStandardSystem_H
