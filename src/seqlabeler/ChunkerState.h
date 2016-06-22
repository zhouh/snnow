/*************************************************************************
	> File Name: ChunkerState.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 15 Jun 2016 12:28:40 PM CST
 ************************************************************************/
#ifndef SNNOW_CHUNKERSTATE_H
#define SNNOW_CHUNKERSTATE_H

#include <algorithm>
#include <iostream>

#include "ChunkerMacro.h"

#include "base/State.h"
#include "SeqLabelerInput.h"
#include "ChunkerAction.h"

class ChunkerState : public State {
public:
    int index_;

    std::vector<int> chunked_label_ids_;
    SeqLabelerInput *seq_input_ptr_;

    int seq_len_;

    int ongo_chunk_index_;

    int curr_chunk_index_;

    int curr_head_index_;

    int prev_chunk_index_;

    int prev_head_index_;

public:
    ChunkerState() {
        be_gold  = true;
        index_in_beam = 0;

        clear();
    }

    ~ChunkerState() = default;

    ChunkerState(const ChunkerState &state) {
        this->index_in_beam = state.index_in_beam;
        this->score = state.score;
        this->previous = state.previous;
        this->last_action = state.last_action;
        this->be_gold = state.be_gold;

        this->index_ = state.index_;
        this->seq_len_ = state.seq_len_;
        this->chunked_label_ids_ = state.chunked_label_ids_;
        this->ongo_chunk_index_ = state.ongo_chunk_index_;
        this->curr_chunk_index_ = state.curr_chunk_index_;
        this->curr_head_index_ = state.curr_head_index_;
        this->prev_chunk_index_ = state.prev_chunk_index_;
        this->prev_head_index_ = state.prev_head_index_;
        this->seq_input_ptr_ = state.seq_input_ptr_;
    }

    ChunkerState&operator=(const ChunkerState &state) {
        if (this == &state) {
            return *this;
        }

        this->index_in_beam = state.index_in_beam;
        this->score = state.score;
        this->previous = state.previous;
        this->last_action = state.last_action;
        this->be_gold = state.be_gold;

        this->index_ = state.index_;
        this->seq_len_ = state.seq_len_;
        this->chunked_label_ids_ = state.chunked_label_ids_;
        this->ongo_chunk_index_ = state.ongo_chunk_index_;
        this->curr_chunk_index_ = state.curr_chunk_index_;
        this->curr_head_index_ = state.curr_head_index_;
        this->prev_chunk_index_ = state.prev_chunk_index_;
        this->prev_head_index_ = state.prev_head_index_;
        this->seq_input_ptr_ = state.seq_input_ptr_;

        return *this;
    }

    void setSequenceInput(SeqLabelerInput *seq_input_ptr) {
        this->seq_input_ptr_ = seq_input_ptr;
    }

    bool complete() {
        return index_ == seq_len_ - 1;
    }

    void setBeamIndex(const int index) {
        index_in_beam = index;
    }

    void setSequenceLength(const int len) {
        seq_len_ = len;
    }
    int sequenceLength() {
        return seq_len_;
    }

    void print(std::string filled) const {
        std::cout << filled << "iib: " << index_in_beam << std::endl;
        std::cout << filled << "previous: " << previous << std::endl;
        std::cout << filled << "score: " << score << std::endl;
        std::cout << filled << "gold: " << be_gold << std::endl;
        std::cout << filled << "last action: " << last_action.action_type << " " << last_action.action_label << " " << last_action.action_code << std::endl;
        std::cout << filled << "index: " << index_ << std::endl;
        std::cout << filled << "seq_len_: " << seq_len_ << std::endl;
        std::cout << filled << "ongo_chunk: " << ongo_chunk_index_ << std::endl;
        std::cout << filled << "curr_chunk: " << curr_chunk_index_ << std::endl;
        std::cout << filled << "curr_head: " << curr_head_index_ << std::endl;
        std::cout << filled << "prev_chunk: " << prev_chunk_index_ << std::endl;
        std::cout << filled << "prev_head: " << prev_head_index_ << std::endl;
        std::cout << filled << "chunked labels: " << std::endl;
        std::cout << filled << "  ";
        for (int i = 0; i < chunked_label_ids_.size(); i++) {
            std::cout << i << ":" << chunked_label_ids_[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    void clear() {
        static_cast<ChunkerAction&>(last_action).clear();

        index_ = -1;
        previous = nullptr;
        score = 0.0;

        ongo_chunk_index_ = -1;
        curr_chunk_index_ = -1;
        prev_chunk_index_ = -1;
        curr_head_index_ = -1;
        prev_head_index_ = -1;
    }
};

#endif // SNNOW_CHUNKERSTATE_H
