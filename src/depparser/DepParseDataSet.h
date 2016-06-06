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

        int index = 0;
        while(true){
            DepParseTree tree;
            if( !( is >>  tree ) ){ // if input ends
                break;
            }
            size++;

            DepParseInput input;
            tree.extractInput(input);
            inputs.push_back(input);
            outputs.push_back(tree);
        }
    }


};


#endif //SNNOW_DEPPARSEDATASET_H
