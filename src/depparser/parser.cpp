/*************************************************************************
	> File Name: parser.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<string>
#include<vector>

#include"Depparser.h"
#include"Config.h"

void printStringVector(std::vector<std::string> v){
    for(int i = 0; i < v.size(); i++)
        std::cout<< i << ":"<<v[i]<<"\t";
    std::cout<<std::endl;
}
void printIntVector(std::vector<int> v){
    for(int i = 0; i < v.size(); i++)
        std::cout<< i << ":"<<v[i]<<"\t";
    std::cout<<std::endl;
}
void printDoubleVector(std::vector<double> v){
    for(int i = 0; i < v.size(); i++)
        std::cout<< i << ":"<<v[i]<<"\t";
    std::cout<<std::endl;
}


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
    std::vector<Instance> trainInstances;
    std::vector<Instance> devInstances;

    /*
     * get the gold dev tree
     */
    int index = 0;
    while(true){
        DepTree tree;
        if( !( is_dev >>  tree ) ){ // if input ends
            break;
        }
        DepParseInput input;
        tree.extractInput(input);
        Instance inst(input);
        devInstances.push_back(inst);
        devTrees.push_back(tree);
    }

    /*
     * get the gold training tree
     */
    index = 0;
    while(true){
        DepTree tree;
        if( !( is_train >>  tree ) ){ // if input ends
            break;
        }
        DepParseInput input;
        tree.extractInput(input);
        Instance inst(input);
        trainInstances.push_back(inst);
        goldTrees.push_back(tree);
    }
    std::cout<<"End to Parse!"<<std::endl;

    std::cout<< "First tree!" << std::endl << goldTrees.front()<<std::endl;
    std::cout<< "First instance"<<std::endl;
    trainInstances.front().display();

    // begin to train
    train.train(trainInstances, goldTrees, devInstances, devTrees);
}
