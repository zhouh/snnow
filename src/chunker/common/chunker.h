/*************************************************************************
	> File Name: chunker.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 30 Dec 2015 02:27:29 PM CST
 ************************************************************************/
#ifndef _COMMON_CHUNKER_H_
#define _COMMON_CHUNKER_H_

#define XPU gpu

#define EMBEDDING_XPU_GUIDE 2

#if EMBEDDING_XPU_GUIDE == 1
    #define EMBEDDING_XPU gpu
#elif EMBEDDING_XPU_GUIDE == 2
    #define EMBEDDING_XPU cpu
#endif

typedef double real_t;

#endif
