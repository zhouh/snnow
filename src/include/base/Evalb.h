//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_EVALB_H
#define SNNOW_EVALB_H

#include "Input.h"
#include "Output.h"

class Evalb {

public:
    // return the evaluation score
    virtual double evalb(std::vector<Input>* inputs,
                 std::vector<Output>* predicted_outputs,
                 std::vector<Output>* gold_outputs) = 0;

};


#endif //SNNOW_EVALB_H
