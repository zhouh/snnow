/*************************************************************************
	> File Name: ChunkerTransitionSystem.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 15 Jun 2016 03:49:51 PM CST
 ************************************************************************/
#ifndef SNNOW_CHUNKERTRANSITIONSYSTEM_H
#define SNNOW_CHUNKERTRANSITIONSYSTEM_H

#include <memory>
#include <string>
#include <assert.h>

#include "base/TransitionSystem.h"

#include "ChunkerMacro.h"
#include "ChunkerAction.h"
#include "ChunkerState.h"
#include "HeadWordRule.h"

class ChunkerTransitionSystem : public  TransitionSystem {
public:
    String2IndexMap dep_label_map_ptr_;  // the label map here is the same as in the dictionary
    std::vector<std::string> known_labels_;

    std::shared_ptr<ChunkerActionFactory> action_factory_ptr_;

    int n_begin_;
    int n_inside_;
    int n_outsize_;

    int label_num_;

    std::vector<int> vec_of_labelidx2labeltype_;
    std::vector<int> vec_of_endtype_;
    std::vector<int> vec_of_singletype_;
    std::vector<std::vector<int>> vec_of_actionidx2validactionidxes_;

    std::shared_ptr<HeadWordRule> head_word_rule_ptr_;
public:
    ChunkerTransitionSystem() = default;

    ~ChunkerTransitionSystem() = default;

    void makeTransitions(const std::vector<std::string>& knowLabels, const String2IndexMap& label_map){
        n_begin_ = n_inside_ = n_outsize_ = -1;

        this->known_labels_ = knowLabels;
        this->dep_label_map_ptr_ = label_map;

        label_num_ = known_labels_.size();
        action_factory_ptr_.reset(new ChunkerActionFactory(label_num_));

        vec_of_labelidx2labeltype_.resize(label_num_, -1);

        for (int i = 0; i < label_num_; i++) {
            const std::string &label = knowLabels[i];
            assert (dep_label_map_ptr_.find(label) != dep_label_map_ptr_.end());
            const int label_idx = dep_label_map_ptr_.find(label)->second;

            switch (label[0]) {
                case 'I':
                    n_inside_ = label_idx;
                    break;
                case 'O':
                    n_outsize_ = label_idx;
                    break;
                case 'E':
                    vec_of_endtype_.push_back(label_idx);
                    break;
                case 'S':
                    vec_of_singletype_.push_back(label_idx);
                    break;
                case 'B':
                    n_begin_ = label_idx;
                    break;
                default:
                    std::cerr << "Invalid label: " << label << std::endl;
                    exit(1);
            }
        }

        vec_of_actionidx2validactionidxes_.resize(label_num_, std::vector<int> (label_num_, -1));
        for (int i = 0; i < label_num_; i++) {
            vec_of_actionidx2validactionidxes_[i].resize(label_num_, 0);

            const std::string &label = knowLabels[i];
            const int label_idx = dep_label_map_ptr_.find(label)->second;

            switch (label[0]) {
                case 'I':
                    vec_of_labelidx2labeltype_[label_idx] = ChunkerAction::INSIDE_TYPE;
                    break;
                case 'O':
                    vec_of_labelidx2labeltype_[label_idx] = ChunkerAction::OUTSIDE_TYPE;
                    break;
                case 'E':
                    vec_of_labelidx2labeltype_[label_idx] = ChunkerAction::END_TYPE;
                    break;
                case 'S':
                    vec_of_labelidx2labeltype_[label_idx] = ChunkerAction::SINGLE_TYPE;
                    break;
                case 'B':
                    vec_of_labelidx2labeltype_[label_idx] = ChunkerAction::BEGIN_TYPE;
                    break;
                default:
                    std::cerr << "Invalid label: " << label << std::endl;
                    exit(1);
            }
        }
    }

    void setHeadWordRule(std::shared_ptr<HeadWordRule> head_word_rule_ptr) {
        this->head_word_rule_ptr_ = head_word_rule_ptr;
    }

    void Move(State *state, Action *action) {
        ChunkerAction &chunk_action = static_cast<ChunkerAction&>(*action);

        const int action_type = chunk_action.getActionType();
        ChunkerState &src_state = static_cast<ChunkerState&>(*state);
        ChunkerState &dst_state = static_cast<ChunkerState&>(*(state->previous));

        switch (action_type) {
            case ChunkerAction::BEGIN_TYPE:
                doBeginMove(src_state, dst_state, chunk_action);
                break;
            case ChunkerAction::INSIDE_TYPE:
                doInsideMove(src_state, dst_state, chunk_action);
                break;
            case ChunkerAction::OUTSIDE_TYPE:
                doOutsideMove(src_state, dst_state, chunk_action);
                break;
            case ChunkerAction::END_TYPE:
                doEndMove(src_state, dst_state, chunk_action);
                break;
            case ChunkerAction::SINGLE_TYPE:
                doSingleMove(src_state, dst_state, chunk_action);
                break;
            default:
                std::cerr << "Invalid Move Action Type: " << action->getActionLabel() << std::endl;
                exit(1);
                break;
        }
    }

    /**
     * return the vector of whether the action is unvalid
     */
    void getValidActs(State *state, std::vector<int>& ret_val){
        ChunkerState &chunk_state = static_cast<ChunkerState&>(*state);
        ChunkerAction &last_action = static_cast<ChunkerAction&>(chunk_state.last_action);

        ret_val.clear();

        if (chunk_state.index_ == chunk_state.sequenceLength() - 2) {
            if (last_action.isInitialAction()){
                ret_val.resize(label_num_, -1);

                for (int id : vec_of_singletype_) {
                    ret_val[id] = 0;
                }

                ret_val[n_outsize_] = 0;

                return ;
            }

            const int action_type = last_action.getActionType();
            if (action_type == ChunkerAction::END_TYPE || action_type == ChunkerAction::OUTSIDE_TYPE || action_type == ChunkerAction::SINGLE_TYPE) {
                ret_val.resize(label_num_, -1);

                for (int id : vec_of_singletype_) {
                    ret_val[id] = 0;
                }

                ret_val[n_outsize_] = 0;

                return ;
            }

            if (action_type == ChunkerAction::BEGIN_TYPE || action_type == ChunkerAction::INSIDE_TYPE) {
                ret_val.resize(label_num_, -1);

                for (int id : vec_of_endtype_) {
                    ret_val[id] = 0;
                }

                return ;
            }
        }

        if (last_action.isInitialAction()) {
            ret_val.resize(label_num_, 0);

            ret_val[n_inside_] = -1;
            for (int id : vec_of_endtype_) {
                ret_val[id] = -1;
            }

            return ;
        }

        const int action_type = last_action.getActionType();

        if (action_type == ChunkerAction::BEGIN_TYPE || action_type == ChunkerAction::INSIDE_TYPE) {
            ret_val.resize(label_num_, -1);

            ret_val[n_inside_] = 0;

            for (int id : vec_of_endtype_) {
                ret_val[id] = 0;
            }
        } else {
            ret_val.resize(label_num_, 0);

            ret_val[n_inside_] = -1;
            for (int id : vec_of_endtype_) {
                ret_val[id] = -1;
            }
        }
    }

    Action* StandardMove(State *state, Output *output) {
        ChunkerState &chunk_state = static_cast<ChunkerState&>(*state);
        SeqLabelerOutput &chunk_output = static_cast<SeqLabelerOutput&>(*output);

        if (chunk_state.complete()) {
            std::cerr << "The chunking state is completed!" << std::endl;
            exit(1);
        }

        int index = chunk_state.index_;

        const int label_idx = chunk_output.outputAt(index + 1);
        const int action_type = vec_of_labelidx2labeltype_[label_idx];

        Action* ret = action_factory_ptr_->makeAction(action_type, label_idx);

        return ret;
    }

    void GenerateOutput(State *state, Input *input, Output *output) {

    }

    void StandardMoveStep(State *state, Output *output) {
        if (static_cast<ChunkerState*>(state)->complete()) {
            std::cerr << "The chunking state is completed!" << std::endl;
            exit(1);
        }

        Action* move_action = StandardMove(state, output);

        Move(state, move_action);
    }

private:
    void doBeginMove(ChunkerState &src_state, ChunkerState &dst_state, ChunkerAction &action) {
        dst_state.index_ = src_state.index_ + 1;
        dst_state.previous = &src_state;

        const int label_idx = action.getActionLabel();
        const int action_type = action.getActionType();
        dst_state.chunked_label_ids_.push_back(label_idx);
        if (dst_state.ongo_chunk_index_ == -1) {
            dst_state.ongo_chunk_index_ = 0;
        } else {
            dst_state.prev_chunk_index_ = dst_state.curr_chunk_index_;
            dst_state.curr_chunk_index_ = dst_state.ongo_chunk_index_;
            dst_state.ongo_chunk_index_ = dst_state.index_;
        }

        if (dst_state.curr_chunk_index_ - 1 < 0) {
            dst_state.prev_head_index_ = -1;
        } else {
            dst_state.prev_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.prev_chunk_index_,
                    dst_state.curr_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.curr_chunk_index_ - 1]);
        }
        if (dst_state.ongo_chunk_index_ - 1 < 0) {
            dst_state.curr_head_index_ = -1;
        } else {
            dst_state.curr_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.curr_chunk_index_,
                    dst_state.ongo_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.ongo_chunk_index_ - 1]);
        }
        dst_state.last_action = static_cast<Action&>(action);
        dst_state.seq_len_ = src_state.seq_len_;
    }

    void doInsideMove(ChunkerState &src_state, ChunkerState &dst_state, ChunkerAction &action) {
        dst_state.index_ = src_state.index_ + 1;
        dst_state.previous = &src_state;

        const int label_idx = action.getActionLabel();
        const int action_type = action.getActionType();
        dst_state.chunked_label_ids_.push_back(label_idx);

        dst_state.last_action = static_cast<Action&>(action);
        dst_state.seq_len_ = src_state.seq_len_;
    }

    void doOutsideMove(ChunkerState &src_state, ChunkerState &dst_state, ChunkerAction &action) {
        dst_state.index_ = src_state.index_ + 1;
        dst_state.previous = &src_state;

        const int label_idx = action.getActionLabel();
        const int action_type = action.getActionType();
        dst_state.chunked_label_ids_.push_back(label_idx);
        if (dst_state.ongo_chunk_index_ == -1) {
            dst_state.ongo_chunk_index_ = 0;
        } else {
            dst_state.prev_chunk_index_ = dst_state.curr_chunk_index_;
            dst_state.curr_chunk_index_ = dst_state.ongo_chunk_index_;
            dst_state.ongo_chunk_index_ = dst_state.index_;
        }

        if (dst_state.curr_chunk_index_ - 1 < 0) {
            dst_state.prev_head_index_ = -1;
        } else {
            dst_state.prev_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.prev_chunk_index_,
                    dst_state.curr_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.curr_chunk_index_ - 1]);
        }
        if (dst_state.ongo_chunk_index_ - 1 < 0) {
            dst_state.curr_head_index_ = -1;
        } else {
            dst_state.curr_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.curr_chunk_index_,
                    dst_state.ongo_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.ongo_chunk_index_ - 1]);
        }
        dst_state.last_action = static_cast<Action&>(action);
        dst_state.seq_len_ = src_state.seq_len_;
    }

    void doEndMove(ChunkerState &src_state, ChunkerState &dst_state, ChunkerAction &action) {
        dst_state.index_ = src_state.index_ + 1;
        dst_state.previous = &src_state;

        const int label_idx = action.getActionLabel();
        const int action_type = action.getActionType();
        dst_state.chunked_label_ids_.push_back(label_idx);

        dst_state.last_action = static_cast<Action&>(action);
        dst_state.seq_len_ = src_state.seq_len_;
    }

    void doSingleMove(ChunkerState &src_state, ChunkerState &dst_state, ChunkerAction &action) {
        dst_state.index_ = src_state.index_ + 1;
        dst_state.previous = &src_state;

        const int label_idx = action.getActionLabel();
        const int action_type = action.getActionType();
        dst_state.chunked_label_ids_.push_back(label_idx);
        if (dst_state.ongo_chunk_index_ == -1) {
            dst_state.ongo_chunk_index_ = 0;
        } else {
            dst_state.prev_chunk_index_ = dst_state.curr_chunk_index_;
            dst_state.curr_chunk_index_ = dst_state.ongo_chunk_index_;
            dst_state.ongo_chunk_index_ = dst_state.index_;
        }

        if (dst_state.curr_chunk_index_ - 1 < 0) {
            dst_state.prev_head_index_ = -1;
        } else {
            dst_state.prev_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.prev_chunk_index_,
                    dst_state.curr_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.curr_chunk_index_ - 1]);
        }
        if (dst_state.ongo_chunk_index_ - 1 < 0) {
            dst_state.curr_head_index_ = -1;
        } else {
            dst_state.curr_head_index_ = head_word_rule_ptr_->findHeadPosition(
                    dst_state.seq_input_ptr_->tag_cache_,
                    dst_state.curr_chunk_index_,
                    dst_state.ongo_chunk_index_ - 1,
                    dst_state.chunked_label_ids_[dst_state.ongo_chunk_index_ - 1]);
        }
        dst_state.last_action = static_cast<Action&>(action);
        dst_state.seq_len_ = src_state.seq_len_;
    }

private:
    ChunkerTransitionSystem(const ChunkerTransitionSystem &) = delete;
    ChunkerTransitionSystem& operator= (const ChunkerTransitionSystem &) = delete;
};

#endif // SNNOW_CHUNKERTRANSITIONSYSTEM_H
