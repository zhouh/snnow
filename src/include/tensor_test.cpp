/*************************************************************************
	> File Name: tensor_test.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 17 Dec 2015 07:12:33 PM CST
 ************************************************************************/
#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;

using namespace mshadow::expr;
using namespace std;

int main(void) {
    float data[4] = {0, 1, 2, 3};

    Tensor<cpu, 2> ts1(data, Shape2(2, 2));
    Tensor<cpu, 2> ts2(data, Shape2(2, 2));

    ts1[0][1] = 4;

    for (int ii = 0; ii < ts2.shape_[0]; ii++) {
        for (int jj = 0; jj < ts2.shape_[1]; jj++) {
            cout << ts2[ii][jj] << " ";
        }
        cout << endl;
    }

    return 0;
}
