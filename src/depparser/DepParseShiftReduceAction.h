//
// Created by zhouh on 16-4-12.
//

#ifndef SNNOW_DEPPARSESHIFTREDUCEACTION_H
#define SNNOW_DEPPARSESHIFTREDUCEACTION_H

#include <vector>
#include <iostream>

#include "base/Action.h"
#include "Dict.h"

class DepParseAction : public Action{

public:
    const static int shift_type = 0;
    const static int left_type = 1;
    const static int right_type = 2;

    DepParseAction(){}

    DepParseAction(int action_type, int action_label){
        this->action_type = action_type;
        this->action_label = action_label;
        action_code = generateActionCode(action_type, action_label);
    }

    DepParseAction(const DepParseAction &action){
        this->action_type = action.action_type;
        this->action_label = action.action_label;
        action_code = action.action_code;
    }

    DepParseAction&operator=(const DepParseAction &action) {
        if (this == &action) {
            return *this;
        }

        this->action_type = action.action_type;
        this->action_label = action.action_label;
        action_code = action.action_code;

        return *this;
    }

    /**
     * get the action code given the action type and action label
     */
    static int generateActionCode(int action_type, int action_label){
//        std::cout<<"action_label_num\t"<<action_label_num<<std::endl;
        switch (action_type){

            case shift_type : return shift_type;
            case DepParseAction::left_type : return (left_type + action_label);
            case right_type : return (action_label + 1 + action_label_num);

        }

        return -1; // error code for return
    }

    std::string toString(std::shared_ptr< Dictionary > dict_ptr){
        switch (action_type){

            case shift_type : return "S";
            case DepParseAction::left_type : return "L-" +dict_ptr->getString( action_label );
            case right_type : return "R-" + dict_ptr->getString( action_label );

        }
    }

};

class DepParseShiftReduceActionFactory : public ActionFactory {

public:
    DepParseAction* shift_action;
    std::vector<DepParseAction*> left_reduce_actions;
    std::vector<DepParseAction*> right_reduce_actions;

    DepParseShiftReduceActionFactory(int action_label_num){
        this->action_type_num = 3;
        this->action_label_num = action_label_num;
        Action::setActionLabelNum(action_label_num);
//        std::cout<<"action_label_num\t"<<action_label_num<<std::endl;
        total_action_num = 2 * action_label_num + 1;
        action_table.resize(total_action_num, nullptr); // resize the action table

        for(int j = 0; j < action_type_num; j++)
        for(int i = 0; i < action_label_num; i++){
            if(j == DepParseAction::shift_type)
                shift_action = static_cast<DepParseAction*>(makeAction(j, i));
            else if( j == DepParseAction::left_type)
                left_reduce_actions.push_back( static_cast<DepParseAction*>(makeAction(j, i)));
            else if( j == DepParseAction::right_type)
                right_reduce_actions.push_back( static_cast<DepParseAction*>(makeAction(j, i)));

        }
    }

    // return the action given action type and label
    virtual Action* makeAction(int action_type, int action_label) {

        int action_code = DepParseAction::generateActionCode(action_type, action_label);
//        std::cout << "action code in make action:\t" << action_code << std::endl;

//        int action_code = -1;
//
//        switch (action_type){
//            case DepParseAction::shift_type : action_code = DepParseAction::shift_type;
//            case DepParseAction::left_type : action_code = (DepParseAction::left_type + action_label);
//            case DepParseAction::right_type : action_code = (action_label + 1 + ActionFactory::action_label_num);
//        }

//        std::cout<< "type\t" << action_type<<"\t label\t"<<action_label<<"\t"<<"action code: "<<action_code<<std::endl;

        if (action_table[action_code] != nullptr) {
            return action_table[action_code].get();
        }
        else {
            std::shared_ptr<DepParseAction> new_action_ptr(new DepParseAction(action_type, action_label));
            action_table[action_code] = new_action_ptr;

//            if (action_type == DepParseAction::shift_type)
//                shift_action = new_action_ptr.operator*();
//            else if (action_type == DepParseAction::left_type)
//                left_reduce_actions.push_back(new_action_ptr.operator*());
//            else if (action_type == DepParseAction::right_type)
//                right_reduce_actions.push_back(new_action_ptr.operator*());
//            else
//                exit(1);  //it is not a valid action type

            return static_cast<Action *>(new_action_ptr.get());
        }

    }
};


#endif //SNNOW_DEPPARSESHIFTREDUCEACTION_H
