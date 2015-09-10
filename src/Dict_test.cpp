/*
 * CDict_test.cpp
 *
 *  Created on: Jun 28, 2015
 *      Author: zhouh
 */

#include "include/Dict.h"

#include <iostream>

int main(){
	vector<string> list = {"first", "second", "third"};
	Dict d(list);
    

	cout << d.getIndex("second");
}
