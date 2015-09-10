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
		for(unsigned i = 0; i < wordlist.size(); ++i)
			map[wordlist[i]] = index++;
	}

	Dict(){
	}
	virtual ~Dict(){};
	
	/*
	 *   load the word list and fill the para member
	 */
	void load(std::vector<std::string> wordlist){
		words = wordlist;
		int index = 0;
		map.clear();
		for(unsigned i = 0; i < wordlist.size(); ++i)
			map[wordlist[i]] = index++;
	}


	inline int getIndex(std::string word){
		auto got = map.find(word);
		return got == map.end() ? -1 : got->second;
	}
	
	inline std::string getWord(int index){
		if( (unsigned)index >= 0 && index < words.size())
			return words[index];
		else
			return nullptr;
	}
	inline int Size(){
		return words.size();
	}

private:
    std::vector<std::string> words;
    std::unordered_map<std::string, int> map;

};

#endif /* INCLUDE_DICT_H_ */
