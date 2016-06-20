/*************************************************************************
	> File Name: HeadWordRule.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: 2016年03月07日 星期一 21时54分51秒
 ************************************************************************/
#ifndef _CHUNKER_COMMON_HEADWORDRULE_
#define _CHUNKER_COMMON_HEADWORDRULE_
#include <memory>
#include <string>
#include <tr1/unordered_map>

#include "Dict.h"

class HeadWordRule {
public:
    static const std::string ADJP_STR;
    static const int ADJP_ID;
    static const std::string ADVP_STR;
    static const int ADVP_ID;
    static const std::string CONJP_STR;
    static const int CONJP_ID;
    static const std::string INTJ_STR;
    static const int INTJ_ID;
    static const std::string LST_STR;
    static const int LST_ID;
    static const std::string NP_STR;
    static const int NP_ID;
    static const std::string O_STR;
    static const int O_ID;
    static const std::string PP_STR;
    static const int PP_ID;
    static const std::string PRT_STR;
    static const int PRT_ID;
    static const std::string SBAR_STR;
    static const int SBAR_ID;
    static const std::string UCP_STR;
    static const int UCP_ID;
    static const std::string VP_STR;
    static const int VP_ID;

    static const std::vector<std::string> ADJPs; //left
    static const std::vector<std::string> ADVPs; //right
    static const std::vector<std::string> CONJPs;//right
    static const std::vector<std::string> INTJs; //left
    static const std::vector<std::string> LSTs;  //right
    static const std::vector<std::vector<std::string>> NPs;   //
    static const std::vector<std::string> Os;    //default
    static const std::vector<std::string> PPs;   //right
    static const std::vector<std::string> PRTs;  //right
    static const std::vector<std::string> SBARs; //left
    static const std::vector<std::string> UCPs;  //right
    static const std::vector<std::string> VPs;   //left

    std::vector<int> labelIdx2ChunkType;
    std::tr1::unordered_map<std::string, int> label2ChunkType;

    std::vector<int> ADJP_priority;
    std::vector<int> ADVP_priority;
    std::vector<int> CONJP_priority;
    std::vector<int> INTJ_priority;
    std::vector<int> LST_priority;
    std::vector<std::vector<int>> NP_priority;
    std::vector<int> O_priority;
    std::vector<int> PP_priority;
    std::vector<int> PRT_priority;
    std::vector<int> SBAR_priority;
    std::vector<int> UCP_priority;
    std::vector<int> VP_priority;
public:
    HeadWordRule(const std::shared_ptr<Dictionary> &posDict, const std::shared_ptr<Dictionary> &labelDict) {
        init(posDict, labelDict);
    }
    ~HeadWordRule() {}

    int findHeadPosition(const std::vector<int> &POSs, const int start, const int end, const int labelIdx);

private:
    void init(const std::shared_ptr<Dictionary> &posDict, const std::shared_ptr<Dictionary> &labelDict);

    int findHeadInADJP(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInADVP(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInCONJP(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInLST(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInNP(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInPP(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInPRT(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInSBAR(const std::vector<int> &POSs,const int start, const int end);
    int findHeadInVP(const std::vector<int> &POSs,const int start, const int end);
};

#endif
