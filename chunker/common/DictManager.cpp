/*************************************************************************
	> File Name: DictManager.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 02:39:54 PM CST
 ************************************************************************/
#include "DictManager.h"

const std::string DictManager::WORDDESCRIPTION = "word";
const std::string DictManager::POSDESCRIPTION = "pos";
const std::string DictManager::LABELDESCRIPTION = "label";
const std::string DictManager::CAPDESCRIPTION  = "capital";

void DictManager::init(const ChunkedDataSet &goldSet) {
    m_mStr2Dict[WORDDESCRIPTION] = std::shared_ptr<Dictionary>(new WordDictionary());
    m_mStr2Dict[POSDESCRIPTION] = std::shared_ptr<Dictionary>(new POSDictionary());
    m_mStr2Dict[LABELDESCRIPTION] = std::shared_ptr<Dictionary>(new LabelDictionary());
    m_mStr2Dict[CAPDESCRIPTION] = std::shared_ptr<Dictionary>(new CapitalDictionary());

    makeDictionaries(goldSet);
#ifdef DEBUG
    std::cerr << "Label dictionary: " << std::endl;
    m_mStr2Dict[LABELDESCRIPTION]->printDict();
#endif
}

void DictManager::saveDictManager(std::ostream &os) {
    os << "dictSize" << " " << m_mStr2Dict.size() << std::endl;

    for (auto &it : m_mStr2Dict) {
        os << it.first << std::endl;
        it.second->saveDictionary(os);
    }
}

void DictManager::loadDictManager(std::istream &is) {
    std::string line;

    getline(is, line);
    int size;
    std::string tmp;
    std::istringstream iss(line);
    iss >> tmp >> size;

    for (int i = 0; i < size; i++) {
        getline(is, line);
        std::istringstream dictNameIss(line);
        std::string dictName;
        dictNameIss >> dictName;
        std::shared_ptr<Dictionary> dict;
        if (dictName == WORDDESCRIPTION) {
            dict.reset(new WordDictionary());
        } else if (dictName == POSDESCRIPTION) {
            dict.reset(new POSDictionary());
        } else if (dictName == CAPDESCRIPTION) {
            dict.reset(new CapitalDictionary());
        } else if (dictName == LABELDESCRIPTION) {
            dict.reset(new LabelDictionary());
        } else {
            std::cerr << "load wrong dictionary: " << dictName << std::endl;
            exit(0);
        }

        dict->loadDictionary(is);
        m_mStr2Dict[dictName] = dict;
    }
}
