//
// Created by zhouh on 16-4-11.
//

#ifndef SNNOW_DEPPARSEDATASET_H
#define SNNOW_DEPPARSEDATASET_H

#include <iostream>
#include <memory>
#include <fstream>

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

//        int index = 0;
        while(true){
            DepParseTree* tree_ptr = new DepParseTree;
            if( !( is >>  tree_ptr ) ){ // if input ends
                break;
            }
            size++;

            DepParseInput* input_ptr = new DepParseInput;
            tree_ptr->extractInput(*input_ptr);
            inputs.push_back(static_cast<Input*>(input_ptr));
            outputs.push_back(static_cast<Output*>(tree_ptr));
        }
    }

    ~DepParseDataSet(){
        for(int i = 0; i < inputs.size(); i++){
            delete inputs[i];
            delete outputs[i];
        }

    }


};


#endif //SNNOW_DEPPARSEDATASET_H
