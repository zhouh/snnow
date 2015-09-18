/*************************************************************************
	> File Name: parser.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include<iostream>
#include<fstream>

#include"Depparser.h"
#include"Config.h"

int main(int argc, char* argv[]){
    std::cout<<"Begin to Parse!"<<std::endl;

    //CConfig::ReadConfig( argv[1] );
    Depparser train(true);

    /*
     * Prepare the input 
     */
    std::ifstream is_train(CConfig::strTrainPath.c_str());
    std::ifstream is_dev(CConfig::strdevPath.c_str());
    std::vector<DepTree> goldTrees;
    std::vector<DepTree> devTrees;
    std::vector<DepParseInput> trainInputs;
    std::vector<DepParseInput> devInputs;

    /*
     * get the gold dev tree
     */
    int index = 0;
    while(true){
        std::cout<<index++<<std::endl; 
        DepTree tree;
        if( !( is_dev >>  tree ) ){ // if input ends
            break;
        }
        DepParseInput input;
        tree.extractInput(input);
        devInputs.push_back(input);
        devTrees.push_back(tree);
    }

    /*
     * get the gold training tree
     */
    index = 0;
    while(true){
        std::cout<<index++<<std::endl; 
        DepTree tree;
        if( !( is_train >>  tree ) ){ // if input ends
            break;
        }
        DepParseInput input;
        tree.extractInput(input);
        trainInputs.push_back(input);
        goldTrees.push_back(tree);
    }
    std::cout<<"End to Parse!"<<std::endl;

    std::cout<< "First tree!" << std::endl << goldTrees.front()<<std::endl;

    // begin to train
    train.train(trainInputs, goldTrees, devInputs, devTrees);
}
