/*************************************************************************
	> File Name: BeamExample.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 16 Jan 2016 01:23:51 PM CST
 ************************************************************************/
#ifndef _CHUNKER_BEAMCHUNKER_BEAMEXAMPLE_H_
#define _CHUNKER_BEAMCHUNKER_BEAMEXAMPLE_H_

#include "mshadow/tensor.h"

class BeamExample {
public:
    TensorContainer<cpu, 2, real_t> input;
    TensorContainer<cpu, 2, real_t> output;

    BeamExample(TensorContainer<cpu, 2, real_t> &in, TensorContainer<cpu, 2, real_t> &out) : input(in.shape_), output(out.shape_) {
        input = in;
        output = out;
    }

    ~BeamExample() {}
};

#endif
