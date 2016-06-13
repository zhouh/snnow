//
// Created by zhouh on 16-3-29.
//

#ifndef SNNOW_ACTION_H
#define SNNOW_ACTION_H

#include <string>
#include <vector>
#include <hash_map>
#include <memory>

class ActionFactory;


/**
 * The action object in the transition system
 */
class Action{

public:
    int action_code; // each action corresponds to a integer

    // the string for the action type and action label
    // of the given action
    int action_type;
    int action_label;

    static int action_label_num;

    Action() = default;

    Action(int action_type, int action_label){
        this->action_type = action_type;
        this->action_label = action_label;
        action_code = action_label_num * action_type + action_label;
    }

    Action(Action & a){
        action_code = a.action_code;
        action_type = a.action_type;
        action_label = a.action_label;
    }

    static void setActionLabelNum(int aln){
        action_label_num = aln;
    }

    // return the action code
    int getActionCode() { return action_code; }


    int getActionType() const { return action_type; }
    int getActionLabel() { return action_label; }

    virtual ~Action() = default;


};


/**
 * Action factory, to generate and store actions.
 *
 */
class ActionFactory{

public:
    // the total number of action type and action label in this system
    int action_type_num;
    int action_label_num;
    int total_action_num;


    // table to store the action in each index, the index is the code of that action
    static std::vector<std::shared_ptr<Action> > action_table;

public:

    ActionFactory() = default;

    // constructor
    // static set the action type num and the action
    ActionFactory(int action_type_num, int action_label_num){
        this->action_type_num = action_type_num;
        this->action_label_num = action_label_num;
        total_action_num = action_type_num * action_label_num;
        action_table.resize(action_label_num * action_type_num, nullptr); // resize the action table
        Action::setActionLabelNum(action_label_num);

    }

    // return the action given action type and label
    virtual Action* makeAction(int action_type, int action_label) {

        int action_code = action_label_num * action_type + action_label;
        if( action_table[action_code] != nullptr){
            return action_table[action_code].get();
        }
        else{
            std::shared_ptr<Action> new_action_ptr( new Action(action_type, action_label));
            action_table[action_code] = new_action_ptr;
            return new_action_ptr.get();
        }

    }

};
#endif //SNNOW_ACTION_H
