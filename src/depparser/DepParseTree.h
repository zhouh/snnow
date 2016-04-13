//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DEPPARSETREE_H
#define SNNOW_DEPPARSETREE_H



#include <vector>
#include <algorithm>
#include <iostream>

#include "DepParseShiftReduceAction.h"
#include "DepTreeNode.h"
#include "base/Output.h"



class DepParseTree : public Output{

public:

    int size; // the size of the tree

    // construct the test DepParseTree, only includes words and tags, but heads, labels
    DepParseTree(DepParseInput& input){

        //add other sentences
        for(auto iter = input.begin(); iter != input.end(); iter++){
            DepParseTreeNode node(iter->first, iter->second);
            nodes.push_back(node);
        }

        size = nodes.size();
    }

    // empty tree, for reading data
    DepParseTree(){
        size = 0;
    }

    void init(DepParseInput & input){
        //add other sentences
        for(auto iter = input.begin(); iter != input.end(); iter++){
            DepParseTreeNode node(iter->first, iter->second);
            nodes.push_back(node);
        }
        size = nodes.size();
    }
    ~DepParseTree(){}

    void extractInput(DepParseInput* input){
        if(size == 0) {
            std::cerr<<" the input dependency tree null!";
            exit(0);
        }

        input.resize(size);

        for(int i = 0; i < size; ++i){
            input[i].first = nodes[i].word;
            input[i].second = nodes[i].tag;
        }
    }

    inline void setHead(int child, int head){ nodes[child].head = head; }
    inline void setLabel(int child, std::string label){ nodes[child].label = label; }

public:
    //para member
    std::vector<DepTreeNode> nodes;

};

// input the gold DepParseTree
inline std::istream & operator >> (std::istream &is, DepParseTree &tree) {

    DepParseTreeNode rootnode(root, root, -1, root); // add root node
    tree.nodes.push_back(rootnode);

    std::string line;
    std::getline(is, line);

    while( is && !line.empty() ){  //not empty line
        DepParseTreeNode node;
        std::istringstream iss(line);
        iss >> node;
        tree.nodes.push_back( node );
        std::getline( is, line );
    }

    tree.size = tree.nodes.size();
    return is ;
}

inline std::ostream & operator << (std::ostream &os, const DepParseTree &tree) {

    // output from node 1, skip root node 
    for(unsigned i = 0; i < tree.nodes.size(); i++)
        os << tree.nodes[i];
    os << std::endl;

    return os ;
}


#endif //SNNOW_DEPPARSETREE_H
