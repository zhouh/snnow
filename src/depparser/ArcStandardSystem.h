/*************************************************************************
  > File Name: src/depparser/ArcStandardSystem.h
  > Author: Hao Zhou
  > Mail: haozhou0806@gmail.com 
  > Created Time: 20/10/15 14:41:46
 ************************************************************************/
#ifndef ARCSTANDARD_SYSTEM_H
#define ARCSTANDARD_SYSTEM_H 

#include<string>
#include<vector>
#include"assert.h"

#include "DepAction.h"
#include "State.h"
/**
 * this is a state pattern
 */
class ArcStandardSystem{

    public:
        std::vector<std::string> knowLabels;
        int rootLabelIndex = -1;
        int nShift;
        int nLeftFirst;
        int nRightFirst;
        int nActNum;

        ArcStandardSystem(){
        }

        void makeTransition(std::vector<std::string> & knowLabels){
            this->knowLabels = knowLabels;
            for(int i = 0; i < knowLabels.size(); i++)
                if(knowLabels[i] == "ROOT"
                    || knowLabels[i] == "root")
                    rootLabelIndex = i;

            nShift = 0; 
            nLeftFirst = 1;
            nActNum = 2 * knowLabels.size() + 1; // sum over 2*n+1 actions
            nRightFirst = knowLabels.size() + nLeftFirst;
        }

        inline int getRootLabelIndex(){
            assert(rootLabelIndex != -1);
            return rootLabelIndex;
        }

        inline int getLeftFirst(){
            return nLeftFirst;
        }

        inline int getRightFirst(){
            return nRightFirst;
        }

        inline int getShift(){
            return nActNum;
        }

        /**
         *   return the action code
         */
        inline unsigned EncodeAction(const unsigned action,
                const unsigned & label = 0) {

            if (action == nShift)
                return action;
            else
                return action + label;
        }

        /**
         *   get the action type
         */
        inline unsigned DecodeUnlabeledAction(const unsigned & action) {
            assert(action < nActNum);

            if (action < nLeftFirst)
                return nShift;
            else if (action < nRightFirst)
                return nLeftFirst;
            else
                return nRightFirst;

        }

        /**
         *  get the dependency label ID
         */
        inline unsigned DecodeLabel(const unsigned & action) {
            assert(action < nActNum);
            return action - DecodeUnlabeledAction(action);
        }

        void displayValidActions(std::vector<int> & validActs){
            for(int i = 0; i < validActs.size(); i++){
                if(validActs[i] < 0)
                    continue;
                if(DecodeUnlabeledAction(i) == nShift)
                    std::cout<<"Shift "; 
                if(DecodeUnlabeledAction(i) == nLeftFirst) 
                    std::cout<<"kArcLeftFirst "; 
                if(DecodeUnlabeledAction(i) == nRightFirst) 
                    std::cout<<"kArcRightFirst ";
                std::cout<< knowLabels[DecodeLabel(i)]<<std::endl;
            }
        }


        /*
         * Perform Arc-Left operation in the arc-standard algorithm
         */
        void ArcLeft(State & state, int label);

        /*
         * Perform the arc-right operation in arc-standard
         */
        void ArcRight(State & state, int label);

        /* 
         * the shift action does pushing
         */
        void Shift(State &state);

        void Move(State & state, const int action);

        void getValidActs(State & state, std::vector<int> & retval);

        int StandardMove(State & state, const DepTree & tree, const std::vector<int> & labelIndexs);

        void GenerateOutput(const State & state, const DepParseInput &input, DepTree &output);

        void StandardMoveStep(State & state, const DepTree & tree, const std::vector<int> & labelIndexs);
};

#endif
