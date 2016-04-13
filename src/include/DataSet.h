//
// Created by zhouh on 16-4-5.
//

#ifndef SNNOW_DATASET_H
#define SNNOW_DATASET_H

#include <vector>
#include <string>
#include <memory>

#include "base/Input.h"
#include "base/Output.h"

/**
 * Data set includes inputs and outputs
 */
class DataSet {

public:

    int size;

    std::vector<std::shared_ptr<Input>>& inputs;
    std::vector<std::shared_ptr<Output>>& outputs;

    /**
     * construct the data set from file stream
     */
    DataSet(std::string file_name) = 0;

    int getSize(){ return size; }
};


#endif //SNNOW_DATASET_H
