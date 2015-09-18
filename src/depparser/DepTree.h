/*
 * DepTree.h
 *
 *  Created on: Jul 2, 2015
 *      Author: zhouh
 */

#ifndef DEPPARSER_DEPTREE_H_
#define DEPPARSER_DEPTREE_H_

#include <vector>
#include <algorithm>
#include <iostream>

#include "DepTreeNode.h"
#include "DepAction.h"

typedef std::vector<std::pair<std::string, std::string>> DepParseInput;

class DepTree{

public:

	int size;

	// construct the test DepTree, only includes words and tags, but heads, labels
	DepTree(DepParseInput input){

		//add other sentences
        for(auto iter = input.begin(); iter != input.end(); iter++){
            DepTreeNode node(iter->first, iter->second);
			nodes.push_back(node);
        }
	}

    // empty tree, for reading data
	DepTree(){
		size = 0;
	}
	~DepTree(){}

    void extractInput(DepParseInput& input){
        if(size == 0) {
            std::cerr<<" the input dependency tree null!";
            exit(0);
        }

        input.resize(size);

        for(int i = 1; i < size; ++i){
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

// input the gold DepTree
	inline std::istream & operator >> (std::istream &is, DepTree &tree) {

		// add root node
		DepTreeNode rootnode(root, root);
		tree.nodes.push_back(rootnode);

        std::string line;
        std::getline(is, line);

		while( is && !line.empty() ){  //not empty line
            DepTreeNode node;
            std::istringstream iss(line);
            iss >> node;
			tree.nodes.push_back( node );
            std::getline( is, line );
		}

		tree.size = tree.nodes.size();
		return is ;
	}

	inline std::ostream & operator << (std::ostream &os, const DepTree &tree) {

        // output from node 1, skip root node 
		for(unsigned i = 1; i < tree.nodes.size(); i++)
			os << tree.nodes[i];
		os << std::endl;

		return os ;
	}
#endif /* DEPPARSER_DEPTREE_H_ */
