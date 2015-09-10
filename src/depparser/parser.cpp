/*************************************************************************
	> File Name: parser.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include<iostream>

#include"Depparser.h"
#include"Config.h"

int main(int argc, char* argv[]){
    CConfig::ReadConfig( argv[1] );
    Depparser train(true);
   
}



