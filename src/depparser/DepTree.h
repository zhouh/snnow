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

#include "DepTreeNode.h"

typedef std::vector<std::pair<std::string, std::string>> DepParseInput;

class DepTree{

public:

	int size;

	// construct the test DepTree, only includes words and tags, but heads, labels
	DepTree(DepParseInput input){
		size = input.size() + 1;

		//add root node
		DepTreeNode node(root, root);
		nodes.push_back(node);

		//add other sentences
		for_each (input.begin(), input.end(), [nodes](std::pair inputPair){
			DepTreeNode node(inputPair.first, inputPair.second);
			nodes.push_back(node);
		});
	}
	DepTree(){
		size = 0;
	}
	~DepTree(){}

	inline void setHead(int child, int head){ nodes[child].head = head; }
	inline void setLabel(int child, std::string label){ nodes[child].label = label; }

	// input the gold DepTree
	inline std::istream & operator >> (std::istream &is, DepTree &tree) {

		// add root node
		DepTreeNode rootnode(root, root);
		nodes.push_back(rootnode);

		DepTreeNode node();
		is >> node;

		while(node.head != -1){  //not empty line
			nodes.push_back(node);
			is >> node;
		}

		size = nodes.size();
		return is ;
	}

	inline std::ostream & operator << (std::ostream &os, const DepTree &tree) {

		for(int i = 0; i < nodes.size(); i++)
			os << nodes[i];
		os << std::endl;

		return os ;
	}

private:
	//para member
	std::vector<DepTreeNode> nodes;

};

#endif /* DEPPARSER_DEPTREE_H_ */
