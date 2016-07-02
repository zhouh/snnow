//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_DepArcStandardSystem_H
#define SNNOW_DepArcStandardSystem_H

#include <memory>
#include <string>

#include "DepParseMacro.h"
#include "base/TransitionSystem.h"
#include "DepParseShiftReduceAction.h"
#include "DepParseState.h"
#include "DepParseMacro.h"

class DepArcStandardSystem : public TransitionSystem{

public:

    String2IndexMap dep_label_map_ptr;  // the label map here is the same as in the dictionary
    std::vector<std::string> known_labels;

    int rootLabelIndex = -1;
    static std::string c_root_str;


    std::shared_ptr<DepParseShiftReduceActionFactory> action_factory_ptr;

    DepArcStandardSystem() {
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
    void makeTransition(const std::vector<std::string>& knowLabels, const String2IndexMap& label_map){

        /*
         * construct the dict for dep label by the know label set
         */
        this->known_labels = knowLabels;
        dep_label_map_ptr =label_map;

        std::cout<< "know label size:\t"<<knowLabels.size()<<std::endl;
        action_factory_ptr.reset(new DepParseShiftReduceActionFactory( knowLabels.size() ));

        // get the root label index
        for (int i = 0; i < known_labels.size(); ++i) {
            if(known_labels[i] == c_root_str)
                rootLabelIndex = i;
        }


    }

    static void setRootLabelStr(std::string root) {
        c_root_str = root;
    }

    inline int getRootLabelIndex(){
        assert(rootLabelIndex != -1);
        return rootLabelIndex;
    }

    inline std::vector<DepParseAction*>& getLeftActions(){
        return action_factory_ptr->left_reduce_actions;
    }

    inline std::vector<DepParseAction*>& getRightActions(){
        return action_factory_ptr->right_reduce_actions;
    }

    inline DepParseAction* getShiftAction(){
        return action_factory_ptr->shift_action;
    }

    /**
     * action function for state
     *
     * all the action functions are directly functioned on the state itself, instead of
     * return a new object.
     */
    void ArcLeft(DepParseState & state, DepParseAction& action);

    void ArcRight(DepParseState & state, DepParseAction& action);

    void Shift(DepParseState &state);

    //----------------------------------------------------


    /**
     * move according to the given action
     */
    void Move(State * state, Action* action);

    /**
     * According to state, fill the data of valid action info in ret_val
     */
    void getValidActs(State * state, std::vector<int>& ret_val);

    /**
     * return the gold action ptr, given context.
     *
     */
    Action* StandardMove(State* state, Output* tree);

    /**
     * generate outputs in output
     */
    void GenerateOutput(State* state, Input* input, Output* output);

    /**
     * move state to next gold state standardly.
     * Note that here state must be a gold state.
     */
    void StandardMoveStep(State* state, Output* tree);
};
#endif //SNNOW_DepArcStandardSystem_H
