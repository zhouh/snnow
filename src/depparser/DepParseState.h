/*
 * DepParseState.h
 *
 *  Created on: Jun 28, 2015, from ZPar
 *      Author: zhouh
 */

#ifndef SNNOW_DEPPARSER_ARC_STANDARD_STATE_H
#define SNNOW_DEPPARSER_ARC_STANDARD_STATE_H

#include <algorithm>
#include <iostream>
#include "assert.h"

#include "base/State.h"
#include "DepParseTree.h"


class DepParseState : public State{
    public:
        //! stack of words that are currently processed
        std::vector<int> m_Stack;

        //! index for the next word
        int m_nNextWord;

        // the lexical head for each word
        std::vector<int> m_lHeads;

        //! the leftmost dependency for each word (just for cache, temporary info)
        std::vector<int> m_lDepsL;

        //! the rightmost dependency for each word (just for cache, temporary info)
        std::vector<int> m_lDepsR;

        //! the second-leftmost dependency for each word
        std::vector<int> m_lDepsL2;

        //! the second-rightmost dependency for each word
        std::vector<int> m_lDepsR2;

        //! the label of each dependency arc
        std::vector<int> m_lLabels;

        //! the length of the sentence, it's set manually.
        int len_;

    public:
        // constructors and destructor
        DepParseState() {
            be_gold = true; // initial DepParseState is true DepParseState
            //initially, only push root (0) into the stack
            clear();
        }

        void initCache(){

            m_lHeads.clear();
            m_lDepsL.clear();
            m_lDepsR.clear();
            m_lDepsL2.clear();
            m_lDepsR2.clear();
            m_lLabels.clear();
            m_lHeads.resize(len_, -1);
            m_lDepsL.resize(len_, -1);
            m_lDepsR.resize(len_, -1);
            m_lDepsL2.resize(len_, -1);
            m_lDepsR2.resize(len_, -1);
            m_lLabels.resize(len_, -1);

        }

        ~DepParseState() {
        }

        void copy(const DepParseState &item) {

            m_Stack = item.m_Stack;
            m_nNextWord = item.m_nNextWord;

            last_action = item.last_action;
            score = item.score;
            len_ = item.len_;
            previous = item.previous;
            be_gold = item.be_gold;
            index_in_beam = item.index_in_beam;

            std::copy_n(item.m_lHeads.begin(), m_nNextWord, m_lHeads.begin());
            std::copy_n(item.m_lDepsL.begin(), m_nNextWord, m_lDepsL.begin());
            std::copy_n(item.m_lDepsR.begin(), m_nNextWord, m_lDepsR.begin());
            std::copy_n(item.m_lDepsL2.begin(), m_nNextWord, m_lDepsL2.begin());
            std::copy_n(item.m_lDepsR2.begin(), m_nNextWord, m_lDepsR2.begin());
            std::copy_n(item.m_lLabels.begin(), m_nNextWord, m_lLabels.begin());
        }

        //set Neural Net for back propagation
        inline void setBeamIdx(int idx){
            index_in_beam  = idx;
        }

        //! comparison
        inline bool higher(const DepParseState &item) const {
            return score > item.score;
        }

        inline bool equal(const DepParseState &item) const {
            return ((this) == &item);
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
            assert( (unsigned)id < m_Stack.size());
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

        inline int leftLeftDep(const int & id) const{
            if(id == -1)
                return -1;
            int left = leftdep(id);
            if(left == -1)
                return -1;
            return leftdep(left);
        }

        inline int rightRightDep(const int & id) const{
            if(id == -1)
                return -1;
            int right = rightdep(id);
            if(right == -1)
                return -1;
            return rightdep(right);
        }

        inline int size() const {
            return m_nNextWord;
        }

        inline bool complete() const {
            return (m_Stack.size() == 1 && m_nNextWord == len_);
        }

        inline int label(const int & id) const {
            assert(id < m_nNextWord);
            if(id == -1)
                return -1;
            return m_lLabels[id];
        }

        inline void popStack(){
            m_Stack.pop_back();
        }

        inline void pushStack(int item){
            m_Stack.push_back(item);
        }

        inline void setStackTop(int & top) {
            assert(m_Stack.size() >= 1);
            m_Stack[m_Stack.size() - 1] = top;
        }

        void clear() {
            m_nNextWord = 0;
            m_Stack.clear();
            score = 0;
            previous = nullptr;
//            last_action = nullptr;
            m_Stack.push_back(0); //push the root onto stack
            m_nNextWord = 1;
        }

        bool hasChildOnQueue(int head, DepParseTree & tree){

#ifdef DEBUG
            std::cout<<"nextWord\t"<<m_nNextWord<<"\tLen\t"<<len_<<std::endl;
#endif
            for(int i = m_nNextWord; i < len_; ++i)
                if(tree.nodes[i].head == head)
                    return true;
            return false;
        }

        // the clear next action is used to clear the next word, used
        // with forwarding the next word index
        /*void ClearNext() {*/
            //m_lHeads[m_nNextWord] = empty_arc;
            //m_lDepsL[m_nNextWord] = empty_arc;
            //m_lDepsL2[m_nNextWord] = empty_arc;
            //m_lDepsR[m_nNextWord] = empty_arc;
            //m_lDepsR2[m_nNextWord] = empty_arc;
            //m_lLabels[m_nNextWord] = empty_label;
        //}

        void display(){
            std::cout<<"--------------"<<std::endl;
            std::cout<<"input len: "<<len_<<std::endl;
            std::cout<<"stack: "<<stack2top()<<" "<<stacktop()<<" size: "<<stacksize()<<std::endl;
            std::cout<<"next word "<<m_nNextWord<<std::endl;
            std::cout<<"last action: "<<last_action.getActionCode()<<std::endl;
            std::cout<<"--------------"<<std::endl;
        }

        void dispalyCache(){
            int a, b;
            std::cout<<"====================="<<std::endl;
            a = rightdep( stack2top() );
            b = rightdep( stacktop() );
            std::cout<<"R "<<"top1: "<< a <<"$"<< label(a) << " top0 " << b << "$" <<label(b)<<std::endl;
            a = leftdep( stack2top() );
            b = leftdep( stacktop() );
            std::cout<<"L "<<"top1: "<< a  <<"$"<< label(a) << " top0 " << b  << "$" <<label(b)<<std::endl;
            a = right2dep( stack2top() );
            b = right2dep( stacktop() );
            std::cout<<"R2 "<<"top1: "<< a <<"$"<< label(a) << " top0 " << b << "$" <<label(b)<<std::endl;
            a = left2dep( stack2top() );
            b = left2dep( stacktop() );
            std::cout<<"L2 "<<"top1: "<< a <<"$"<< label(a) << " top0 " << b << "$" <<label(b)<<std::endl;
            a = rightRightDep( stack2top() );
            b = rightRightDep( stacktop() ) ;
            std::cout<<"RR "<<"top1: "<< a <<"$"<< label(a) << " top0 " << b<< "$" <<label(b)<<std::endl;
            a = leftLeftDep( stack2top() );
            b = leftLeftDep( stacktop() );
            std::cout<<"LL "<<"top1: "<< a <<"$"<< label(a) << " top0 " << b << "$" <<label(b)<<std::endl;
            std::cout<<"====================="<<std::endl;

        }

        //void printActionSequence(ArcStandardSystem * tranSystem){
        //int actionSize = 0;
        //DepParseState * DepParseState = this;
        //while( DepParseState!= nullptr){
        //if(DepParseState->last_action == -1)
        //break;
        //actionSize++;
        //std::cout<<DepParseState->last_action<<"\t"<<tranSystem->DecodeUnlabeledAction( DepParseState->last_action )<<"+"<<tranSystem->DecodeLabel( DepParseState->last_action )<<std::endl;
        //DepParseState = DepParseState->previous_;
        //}
        //std::cout<<"senLen"<<len_<<"\tsum\t"<<actionSize<<std::endl;
        //}

};



#endif  //  end for DEPPARSER_ARC_STANDARD_DepParseState_H
