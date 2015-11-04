/*************************************************************************
  > File Name: src/depparser/ArcstandardSystem.cpp
  > Author: Hao Zhou
  > Mail: haozhou0806@gmail.com 
  > Created Time: 16/10/15 10:36:25
 ************************************************************************/

#include "ArcStandardSystem.h"

// the move action is a simple call to do action according to the action code
void ArcStandardSystem::Move(State & state, const int action) {

    int actType = DecodeUnlabeledAction(action);
    if(actType == nShift){
        Shift(state);
        return;
    }
    else if(actType == nLeftFirst){
        ArcLeft(state,  DecodeLabel(action) );
        return;
    }
    else if(actType == nRightFirst){
        ArcRight(state,  DecodeLabel(action) );
        return;
    }
    else{
        std::cerr << "Move Invalid Action Type: " << action<<std::endl;
        exit(1);
    }
}

//-----------------------------------------------------------------------------
void ArcStandardSystem::getValidActs(State & state, std::vector<int> & retval) {

    retval.resize(nActNum, 0);
    retval[nLeftFirst + rootLabelIndex] = -1; //left-root is unvalid
    int stack_size = state.m_Stack.size();
    int queue_size = state.len_ - state.m_nNextWord;

    //shift
    if (queue_size <= 0)
        retval[nShift] = -1;

    int stack_left = state.stack2top();
    //reduce
    if (stack_size >= 2 && state.stack2top()!= 0) { // all reduce is valid except reduce root
        retval[nRightFirst + rootLabelIndex] = -1;
        retval[nLeftFirst + rootLabelIndex] = -1;
        return;
    }
    else {
        for (int i = nLeftFirst; i < nActNum; ++i) //reduce is unvalid
            retval[i] = -1;

        if( state.stack2top() == 0 && queue_size == 0) //except right reduce root
            retval[nRightFirst + rootLabelIndex] = 0;
        return;
    }

    return;
}

int ArcStandardSystem::StandardMove(State & state, const DepTree & tree, const std::vector<int> & labelIndexs) {
    if (state.complete()) {
        std::cerr << "The parsing state is completed!" << std::endl;
        exit(1);
    }

    int w2 = state.stacktop();
    int w1 = state.stack2top();
    int stackSize = state.stacksize();

#ifdef DEBUG
    std::cout<<"w1: "<<w1<<"    w2:  "<<w2<<"   w1 head: "<<tree.nodes[w1].head<<"  stacksize:    "<<stacksize()<<std::endl;
    if(w1 != -1 && w2 != -1){
        std::cout<<"w1 head\t"<<tree.nodes[w1].label<<"\tw2 head\t"<<tree.nodes[w2].label<<std::endl;
        std::cout<<"w1 head\t"<<labelIndexs[w1]<<"\tw2 head\t"<<labelIndexs[w2]<<std::endl;
        std::cout<<"arcRF "<<kArcRightFirst<<std::endl;
    }
#endif
    if( stackSize >= 2 && tree.nodes[w1].head == w2)
        return nLeftFirst + labelIndexs[w1];
    if( stackSize >= 2 && tree.nodes[w2].head == w1 && !state.hasChildOnQueue(w2, tree) )
        return nRightFirst + labelIndexs[w2];
    return nShift;
}

void ArcStandardSystem::StandardMoveStep(State & state, const DepTree & tree, const std::vector<int> & labelIndexs) {
    int action = StandardMove(state, tree, labelIndexs);
    Move(state, action);
}

// we want to pop the root item after the whole tree done
// on the one hand this seems more natural
// on the other it is easier to score
void ArcStandardSystem::GenerateOutput(const State & state, const DepParseInput &input, DepTree &output) {
    std::cout<<"generate tree"<<std::endl;
#ifdef DEBUG
    state.printActionSequence();
    for(int i = 0; i < len_; i++){
        std::cout<<i<<"\tlabel\t"<<state.m_lLabels[i]<<std::endl;
    }
#endif
    for (int i = 1; i < state.len_; ++i) {
        output.setHead(i, state.m_lHeads[i]);
#ifdef DEBUG
        std::cout<<"i:\t"<<i<<std::endl;
        std::cout<<"label id\t"<<state.m_lLabels[i]<<std::endl;
        std::cout<<"label string\t"<<knowLabels[state.m_lLabels[i]]<<std::endl;
#endif
        output.setLabel(i, knowLabels[state.m_lLabels[i]]);
    }

    std::cout<<"generate tree end"<<std::endl;
}

/*
 * Perform Arc-Left operation in the arc-standard algorithm
 */
void ArcStandardSystem::ArcLeft(State & state, int label) {
    // At least, there must be two elements in the stack.
    assert(state.stacksize() > 1);

    int stack_size = state.stacksize();
    int top0 = state.stacktop();
    int top1 = state.stack2top();

    state.popStack();
    state.setStackTop(top0);

    state.m_lHeads[top1] = top0;
    state.m_lLabels[top1] = label;

    if (state.m_lDepsL[top0] == empty_arc) {
        state.m_lDepsL[top0] = top1;
    } else if (top1 < state.m_lDepsL[top0]) {
        state.m_lDepsL2[top0] = state.m_lDepsL[top0];
        state.m_lDepsL[top0] = top1;
    } else if (top1 < state.m_lDepsL2[top0]) {
        state.m_lDepsL2[top0] = top1;
    }

    state.last_action = EncodeAction(nLeftFirst, label);
}

/*
 * Perform the arc-right operation in arc-standard
 */
void ArcStandardSystem::ArcRight(State & state, int label) {

    assert(state.stacksize() > 1);


    int stack_size = state.stacksize();
    int top0 = state.stacktop();
    int top1 = state.stack2top();

    state.popStack();
    state.m_lHeads[top0] = top1;
    state.m_lLabels[top0] = label;

    if (state.m_lDepsR[top1] == empty_arc) {
        state.m_lDepsR[top1] = top0;
    } else if (state.m_lDepsR[top1] < top0) {
        state.m_lDepsR2[top1] = state.m_lDepsR[top1];
        state.m_lDepsR[top1] = top0;
    } else if (state.m_lDepsR2[top1] < top0) {
        state.m_lDepsR2[top1] = top0;
    }

    state.last_action = EncodeAction(nRightFirst, label);
}

/* 
 * the shift action does pushing
 */
void ArcStandardSystem::Shift(State & state) {
    state.m_Stack.push_back(state.m_nNextWord);
    state.m_nNextWord++;
    //state.ClearNext();
    state.last_action = nShift;
}

