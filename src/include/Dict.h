/*
 * Dict.h
 *
 *  Created on: Jun 28, 2015
 *      Author: zhouh
 */

#ifndef INCLUDE_DICT_H_
#define INCLUDE_DICT_H_

#include <string>
#include <vector>
#include <unordered_map>


class Dict {
public:
	Dict(std::vector<std::string> wordlist){
		words = wordlist;
		int index = 0;
		for(int i = 0; i < wordlist.size(); ++i)
			map[wordlist[i]] = index++;
	}

	Dict(){
	}
	virtual ~Dict(){};
	
	/*
	 *   load the word list and fill the para member
	 */
	void load(vector<string> wordlist){
		words = wordlist;
		int index = 0;
		map.clear();
		for(int i = 0; i < wordlist.size(); ++i)
			map[wordlist[i]] = index++;
	}


	inline int getIndex(string word){
		auto got = map.find(word);
		return got == map.end() ? -1 : got->second;
	}
	
	inline string getWord(int index){
		if(index >= 0 && index < words.size())
			return words[index];
		else
			return nullptr;
	}
	inline int Size(){
		return words.size();
	}

private:
	vector<string> words;
	unordered_map<string, int> map;

};

#endif /* INCLUDE_DICT_H_ */
