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

    String2IndexMap dep_label_map_ptr;  // the label map here is the same as in the dictionary
    std::vector<std::string> known_labels;

    int rootLabelIndex = -1;
    static std::string c_root_str;


    std::shared_ptr<DepParseShiftReduceActionFactory> action_factory_ptr;

    DepArcStandardSystem() {
        std::clog << "using the default constructor, which is not valid!" << std::endl;
        exit(0);
    }

//    DepArcStandardSystem(std::vector<std::string>& known_labels) {
//
//        makeTransition(known_labels);
//        this->known_labels = known_labels;
//        action_factory_ptr.reset(new DepParseShiftReduceActionFactory( dep_label_map_ptr.operator*().size() ));
//    }

    /**
     * prepare the transition system given the known dependency labels and dictionary for labels
     * The transition system uses the labels for constructing shift-reduce actions
     */
    void makeTransition(std::vector<std::string>& knowLabels, String2IndexMap& label_map){

        /*
         * construct the dict for dep label by the know label set
         */
        this->known_labels = knowLabels;
        dep_label_map_ptr =label_map;
        action_factory_ptr.reset(new DepParseShiftReduceActionFactory( dep_label_map_ptr.size() ));

        // get the root label index
        for (int i = 0; i < known_labels.size(); ++i) {
            if(known_labels[i] == c_root_str)
                rootLabelIndex = i;
        }


    }

    inline int getRootLabelIndex(){
        assert(rootLabelIndex != -1);
        return rootLabelIndex;
    }

    inline std::vector<DepParseAction>& getLeftActions(){
        return action_factory_ptr->left_reduce_actions;
    }

    inline std::vector<DepParseAction>& getRightActions(){
        return action_factory_ptr->right_reduce_actions;
    }

    inline DepParseAction& getShiftAction(){
        return action_factory_ptr->shift_action;
    }

    void ArcLeft(DepParseState & state, Action& action);

    void ArcRight(DepParseState & state, Action& action);

    void Shift(DepParseState &state);

    void Move(State & state, const Action& action);

    void getValidActs(State & state, std::vector<int>& ret_val);

    Action* StandardMove(State& state, const Output& tree);

    void GenerateOutput(const State& state, const Input& input, Output& output);

    void StandardMoveStep(State& state, const Output& tree);
};
#endif //SNNOW_DepArcStandardSystem_H
