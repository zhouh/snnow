/*************************************************************************
	> File Name: ChunkerAction.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 15 Jun 2016 02:20:35 PM CST
 ************************************************************************/
#ifndef SNNOW_CHUNKERACTION_H
#define SNNOW_CHUNKERACTION_H

#include <vector>

#include "base/Action.h"

class ChunkerAction : public Action {
public:
    const static int BEGIN_TYPE = 0;
    const static int INSIDE_TYPE = 1;
    const static int OUTSIDE_TYPE = 2;
    const static int END_TYPE = 3;
    const static int SINGLE_TYPE = 4;

public:
    ChunkerAction() = default;

    ChunkerAction(const int action_type, const int action_label) {
        this->action_type = action_type;
        this->action_label = action_label;
        this->action_code = action_label;
    }

    ~ChunkerAction() = default;

    ChunkerAction(const ChunkerAction &action) {
        this->action_type = action.action_type;
        this->action_label = action.action_label;
        this->action_code = action.action_code;
    }

    ChunkerAction& operator= (const ChunkerAction &action) {
        if (this == &action) {
            return *this;
        }

        this->action_type = action.action_type;
        this->action_label = action.action_label;
        this->action_code = action.action_code;

        return *this;
    }

    bool isInitialAction() {
        return action_label < 0;
    }

    void clear() {
        this->action_type = -1;
        this->action_label = -1;
        this->action_code = -1;
    }

    static int generateActionCode(const int action_type, const int action_label) {
        return action_label;
    }
};

class ChunkerActionFactory : public ActionFactory {
public:
    ChunkerActionFactory(const int action_label_num) {
        this->action_type_num = 5;
        this->action_label_num = action_label_num;
        this->total_action_num = action_label_num;

        action_table.resize(total_action_num, nullptr);
    }

    ChunkerActionFactory() = default;

    ~ChunkerActionFactory() = default;

    Action* makeAction(const int action_type, const int action_label) {
        const int action_code = ChunkerAction::generateActionCode(action_type, action_label);

        if (action_table[action_code] == nullptr) {
            action_table[action_code].reset(new ChunkerAction(action_type, action_label));
        }

        return action_table[action_code].get();
    }

private:
    ChunkerActionFactory(const ChunkerActionFactory &) = delete;
    ChunkerActionFactory& operator= (const ChunkerActionFactory &) = delete;
};

#endif // SNNOW_CHUNKERACTION_H
