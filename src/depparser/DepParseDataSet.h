//
// Created by zhouh on 16-4-11.
//

#ifndef SNNOW_DEPPARSEDATASET_H
#define SNNOW_DEPPARSEDATASET_H

#include <iostream>
#include <memory>

#include "DataSet.h"
#include "DepParseTree.h"
#include "DepParseInput.h"

/**
 * prepare the data set for the dependency parsing
 */
class DepParseDataSet : public DataSet{

public:

    DepParseDataSet(std::string file_name){

        size = 0;

        std::ifstream is(file_name.c_str());

        int index = 0;
        while(true){
            std::shared_ptr<DepParseTree> tree_ptr;
            if( !( is >>  *tree_ptr ) ){ // if input ends
                break;
            }
            size++;
            std::shared_ptr<DepParseInput> input_ptr;
            tree_ptr->extractInput(input_ptr);
            inputs.push_back(input_ptr);
            outputs.push_back(tree_ptr);
        }
    }


};


#endif //SNNOW_DEPPARSEDATASET_H
