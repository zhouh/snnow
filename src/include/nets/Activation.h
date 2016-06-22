//
// Created by zhouh on 16-4-1.
//

#ifndef SNNOW_ACTIVATION_H
#define SNNOW_ACTIVATION_H

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;


// define sigmoid operation
struct sigmoid{

    MSHADOW_XINLINE static real_t Map(real_t a) {
        return  1.0f/(1.0f+expf(-a));
    }
};

struct cube{
    MSHADOW_XINLINE static real_t Map(real_t a){
        return a * a * a;
    }
};

struct threshold {
    MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
        return a < b ? 1.0f : 0.0f;
    }
};

struct zsigmoid{
    MSHADOW_XINLINE static real_t Map(real_t a) {
        if (a > 0) {
            return  expf(-a) / (1.0 + expf(-a));
        } else {
            return 1.0 - (1.0 / (1.0 + expf(-a)));
        }
    }
};

#endif //SNNOW_ACTIVATION_H
