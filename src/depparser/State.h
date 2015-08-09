/*
 * State.h
 *
 *  Created on: Jun 28, 2015, from ZPar
 *      Author: zhouh
 */

#ifndef DEPPARSER_ARC_STANDARD_STATE_H
#define DEPPARSER_ARC_STANDARD_STATE_H

#define MAX_SENTENCE_SIZE 400;

#include <algorithm>

#include "assert.h"
#include "DepAction.h"

class State {
private:
	//! stack of words that are currently processed
	std::vector<int> m_Stack;

	//! index for the next word
	int m_nNextWord;

	// the lexical head for each word
	int m_lHeads[MAX_SENTENCE_SIZE];

	//! the leftmost dependency for each word (just for cache, temporary info)
	int m_lDepsL[MAX_SENTENCE_SIZE];

	//! the rightmost dependency for each word (just for cache, temporary info)
	int m_lDepsR[MAX_SENTENCE_SIZE];

	//! the second-leftmost dependency for each word
	int m_lDepsL2[MAX_SENTENCE_SIZE];

	//! the second-rightmost dependency for each word
	int m_lDepsR2[MAX_SENTENCE_SIZE];

	//! the set of left tags
	CSetOfTags<int> m_lDepTagL[MAX_SENTENCE_SIZE];

	//! the set of right tags
	CSetOfTags<int> m_lDepTagR[MAX_SENTENCE_SIZE];

	//! the label of each dependency arc
	int m_lLabels[MAX_SENTENCE_SIZE];

public:

	int beamIdx = 0;

	static std::string rootLabel = "root";

	//! score of stack
	double score;

	//! the length of the sentence, it's set manually.
	int len_;

	//! Previous state of the current state
	const State * previous_;

	//! the last stack action
	int last_action;

	//! state is gold?
	bool bGold;

public:
	// constructors and destructor
	State() {
		clear();
	}

	// copy constructor
	State(const State &item) {
			m_Stack = item.m_Stack;
			m_nNextWord = item.m_nNextWord;

			last_action = item.last_action;
			score = item.score;
			len_ = item.len_;
			previous_ = item.previous_;

			std::copy_n(item.m_lHeads, m_nNextWord, m_lHeads);
			std::copy_n(item.m_lDepsL, m_nNextWord, m_lDepsL);
			std::copy_n(item.m_lDepsR, m_nNextWord, m_lDepsR);
			std::copy_n(item.m_lDepsL2, m_nNextWord, m_lDepsL2);
			std::copy_n(item.m_lDepsR2, m_nNextWord, m_lDepsR2);
			std::copy_n(item.m_lLabels, m_nNextWord, m_lLabels);
	}

	~State() {
	}

public:
	void copy(const State &item) {
				m_Stack = item.m_Stack;
				m_nNextWord = item.m_nNextWord;

				last_action = item.last_action;
				score = item.score;
				len_ = item.len_;
				previous_ = item.previous_;

				std::copy_n(item.m_lHeads, m_nNextWord, m_lHeads);
				std::copy_n(item.m_lDepsL, m_nNextWord, m_lDepsL);
				std::copy_n(item.m_lDepsR, m_nNextWord, m_lDepsR);
				std::copy_n(item.m_lDepsL2, m_nNextWord, m_lDepsL2);
				std::copy_n(item.m_lDepsR2, m_nNextWord, m_lDepsR2);
				std::copy_n(item.m_lLabels, m_nNextWord, m_lLabels);
	}

	//set Neural Net for back propagation
	inline void setBeamIdx(int idx){
		beamIdx  = idx;
	}

	//! comparison
	inline bool higher(const State &item) const {
		return score > item.score;
	}

	inline bool equal(const State &item) const {
		return !((*this) == item);
	}

	inline int stacksize() const {
		return m_Stack.size();
	}

	inline bool stackempty() const {
		return m_Stack.empty();
	}

	inline int stacktop() const {
		if (m_Stack.empty()) {
			return -1;
		}
		return m_Stack.back();
	}

	inline int stack2top() const {
		if (m_Stack.size() < 2) {
			return -1;
		}
		return m_Stack[m_Stack.size() - 2];
	}

	inline int stack3top() const {
		if (m_Stack.size() < 3) {
			return -1;
		}
		return m_Stack[m_Stack.size() - 3];
	}

	inline int stackbottom() const {
		assert(!m_Stack.empty());
		return m_Stack.front();
	}

	inline int stackitem(const int & id) const {
		assert(id < m_Stack.size());
		return m_Stack[id];
	}

	inline int head(const int & id) const {
		assert(id < m_nNextWord);
		return m_lHeads[id];
	}

	inline int leftdep(const int & id) const {
		assert(id < m_nNextWord);
		return m_lDepsL[id];
	}

	inline int rightdep(const int & id) const {
		assert(id < m_nNextWord);
		return m_lDepsR[id];
	}

	inline int left2dep(const int & id) const {
		assert(id < m_nNextWord);
		return m_lDepsL2[id];
	}

	inline int right2dep(const int & id) const {
		assert(id < m_nNextWord);
		return m_lDepsR2[id];
	}

	inline int size() const {
		return m_nNextWord;
	}

	inline bool complete() const {
		return (m_Stack.size() == 1 && m_nNextWord == len_);
	}

	inline int label(const int & id) const {
		assert(id < m_nNextWord);
		return m_lLabels[id];
	}

	void clear() {
		m_nNextWord = 0;
		m_Stack.clear();
		m_Stack.push_back(0); //push the root onto stack
		score = 0;
		previous_ = 0;
		last_action = -1;
		ClearNext();
	}



	bool hasChildOnQueue(int head, DepTree tree){
		for(int i = m_nNextWord; i < len_; ++i)
			if(tree.nodes[i].head == head)
				return true;
		return false;
	}

//-----------------------------------------------------------------------------
public:

	// Perform Arc-Left operation in the arc-standard algorithm
	void ArcLeft(int label) {
		// At least, there must be two elements in the stack.
		assert(m_Stack.size() > 1);

		int stack_size = m_Stack.size();
		int top0 = m_Stack[stack_size - 1];
		int top1 = m_Stack[stack_size - 2];

		m_Stack.pop_back();
		m_Stack.back() = top0;

		m_lHeads[top1] = top0;
		m_lLabels[top1] = label;

		if (m_lDepsL[top0] == empty_arc) {
			m_lDepsL[top0] = top1;
		} else if (top1 < m_lDepsL[top0]) {
			m_lDepsL2[top0] = m_lDepsL[top0];
			m_lDepsL[top0] = top1;
		} else if (top1 < m_lDepsL2[top0]) {
			m_lDepsL2[top0] = top1;
		}

		last_action = EncodeAction(kArcLeftFirst, label);
	}

	// Perform the arc-right operation in arc-standard
	void ArcRight(int label) {

		assert(m_Stack.size() > 1);

		int stack_size = m_Stack.size();
		int top0 = m_Stack[stack_size - 1];
		int top1 = m_Stack[stack_size - 2];

		m_Stack.pop_back();
		m_lHeads[top0] = top1;

		m_lLabels[top0] = label;

		if (m_lDepsR[top1] == empty_arc) {
			m_lDepsR[top1] = top0;
		} else if (m_lDepsR[top1] < top0) {
			m_lDepsR2[top1] = m_lDepsR[top1];
			m_lDepsR[top1] = top0;
		} else if (m_lDepsR2[top1] < top0) {
			m_lDepsR2[top1] = top0;
		}

		last_action = EncodeAction(kArcRightFirst, label);
	}

	// the shift action does pushing
	void Shift() {
		m_Stack.push_back(m_nNextWord);
		m_nNextWord++;
		ClearNext();
		last_action = kShift;
	}

	// the clear next action is used to clear the next word, used
	// with forwarding the next word index
	void ClearNext() {
		m_lHeads[m_nNextWord] = empty_arc;
		m_lDepsL[m_nNextWord] = empty_arc;
		m_lDepsL2[m_nNextWord] = empty_arc;
		m_lDepsR[m_nNextWord] = empty_arc;
		m_lDepsR2[m_nNextWord] = empty_arc;
		m_lLabels[m_nNextWord] = empty_label;
	}

	// the move action is a simple call to do action according to the action code
	void Move(const int action) {
		switch (DecodeUnlabeledAction(action)) {
		case kShift: {
			Shift();
			return;
		}
		case kArcLeftFirst: {
			ArcLeft(action);
			return;
		}
		case kArcRightFirst: {
			ArcRight(action);
			return;
		}
		default: {
			std::cerr << "Move Invalid Action Type: " << action;
			exit(1);
		}
		}
	}

//-----------------------------------------------------------------------------
public:
	std::vector<int> getValidActs(std::vector<int> & retval) {
		retval.resize(kActNum, 0);
		retval[kArcLeftFirst + rootLabelIndex] = -1; //left-root is unvalid
		int stack_size = m_Stack.size();
		int queue_size = len_ - m_nNextWord;

		//shift
		if (queue_size <= 0) {
			retval[kShift] = -1;
		}

		//reduce
		if (stack_size > 2) {
			int stack_left = stack2top();
			if (stack_left != 0) //is root word
				retval[kArcRightFirst + rootLabelIndex] = -1;
			return retval;
		}

		//reduce is unvalid
		for (int i = kArcLeftFirst; i < kActNum; ++i)
			retval[i] = -1;

		return retval;
	}

public:
	int StandardMove(const DepTree & tree, const std::vector<int> & labelIndexs) {
		if (complete()) {
			std::cerr << "The parsing state is completed!" << std::endl;
			exit(1);
		}

		int w2 = stacktop();
		int w1 = stack2top();

		if(tree.nodes[w1].head == w2)
			return kArcLeftFirst + labelIndexs[w1];
		if(tree.nodes[w2].head == w1 && !hasChildOnQueue(w2, tree))
			return kArcRightFirst + labelIndexs[w2];
		return kShift;

	}

	void StandardMoveStep(const DepTree & tree, const std::vector<int> & labelIndexs) {
		int action = StandardMove(tree, labelIndexs);
		Move(action);
	}

	// we want to pop the root item after the whole tree done
	// on the one hand this seems more natural
	// on the other it is easier to score

	void GenerateTree(const DepParseInput &input, DepTree &output,
			std::vector<std::string> knowLabels) const {
		for (int i = 0; i < len_; ++i) {
			output.setHead(i, m_lHeads[i]);
			output.setLabel(i, knowLabels[m_lLabels[i]]);
		}
	}

};

struct CScoredTransition {
	void operator()(State* s, int a, double sc){
		source = s;
		action = a;
		score = sc;
	}
//! The pointer to the source state;
	const State* source;
//! The compile action applied to the source state;
	int action;
//! The resulted in score.
	double score;
};

#endif  //  end for DEPPARSER_ARC_STANDARD_STATE_H
