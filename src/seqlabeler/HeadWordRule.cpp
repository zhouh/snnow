/*************************************************************************
	> File Name: HeadWordRule.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: 2016年03月07日 星期一 22时49分07秒
 ************************************************************************/
#include <iostream>
#include <assert.h>

#include "HeadWordRule.h"

const std::string HeadWordRule::ADJP_STR = "ADJP";
const int HeadWordRule::ADJP_ID = 0;
const std::string HeadWordRule::ADVP_STR = "ADVP";
const int HeadWordRule::ADVP_ID = 1;
const std::string HeadWordRule::CONJP_STR = "CONJP";
const int HeadWordRule::CONJP_ID = 2;
const std::string HeadWordRule::INTJ_STR = "INTJ";
const int HeadWordRule::INTJ_ID = 3;
const std::string HeadWordRule::LST_STR = "LST";
const int HeadWordRule::LST_ID = 4;
const std::string HeadWordRule::NP_STR = "NP";
const int HeadWordRule::NP_ID = 5;
const std::string HeadWordRule::O_STR = "O";
const int HeadWordRule::O_ID = 6;
const std::string HeadWordRule::PP_STR = "PP";
const int HeadWordRule::PP_ID = 7;
const std::string HeadWordRule::PRT_STR = "PRT";
const int HeadWordRule::PRT_ID = 8;
const std::string HeadWordRule::SBAR_STR = "SBAR";
const int HeadWordRule::SBAR_ID = 9;
const std::string HeadWordRule::UCP_STR = "UCP";
const int HeadWordRule::UCP_ID = 10;
const std::string HeadWordRule::VP_STR = "VP";
const int HeadWordRule::VP_ID = 11;

const std::vector<std::string> HeadWordRule::ADJPs = {"NNS", "QP", "NN", "$", "ADVP", "JJ", "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW", "RBR", "RBS", "SBAR", "RB"}; //left
const std::vector<std::string> HeadWordRule::ADVPs = {"RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN"}; //right
const std::vector<std::string> HeadWordRule::CONJPs = {"CC", "RB", "IN"};//right
const std::vector<std::string> HeadWordRule::INTJs = {}; //left
const std::vector<std::string> HeadWordRule::LSTs = {"LST", ":"};  //right
const std::vector<std::vector<std::string>> HeadWordRule::NPs = {{"POS"}, //check
                                                   {"NN", "NNP", "NNPS", "NNS", "NX","POS", "JJR"}, //right
                                                   {"NP"}, //left
                                                   {"$", "ADJP", "PRN"},//right
                                                   {"CD"}, //right
                                                   {"JJ", "JJS", "RB", "QP"}};   //right
const std::vector<std::string> HeadWordRule::Os = {};    //default
const std::vector<std::string> HeadWordRule::PPs = {"IN", "TO", "VBG", "VBN", "RP", "FW"};   //right
const std::vector<std::string> HeadWordRule::PRTs = {"RP"};  //right
const std::vector<std::string> HeadWordRule::SBARs = {"IN", "DT"}; //left
const std::vector<std::string> HeadWordRule::UCPs = {};  //right
const std::vector<std::string> HeadWordRule::VPs = {"TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "VP", "ADJP", "NN", "NNS", "NP"};   //left

void HeadWordRule::init(const std::shared_ptr<Dictionary> &posDict, const std::shared_ptr<Dictionary> &labelDict) {
    const int unkPOSIdx = posDict->getUnkIndex();

    for (const std::string &pos : HeadWordRule::ADJPs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            ADJP_priority.push_back(found);
        }
    }
    
    for (const std::string &pos : HeadWordRule::ADVPs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            ADVP_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::CONJPs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            CONJP_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::LSTs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            LST_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::PPs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            PP_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::PRTs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            PRT_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::SBARs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            SBAR_priority.push_back(found);
        }
    }

    for (const std::string &pos : HeadWordRule::VPs) {
        int found = posDict->getStringIndex(pos);

        if (found != unkPOSIdx) {
            VP_priority.push_back(found);
        }
    }

    for (auto &v : HeadWordRule::NPs) {
        std::vector<int> priority;
        for (const std::string &pos : v) {
            int found = posDict->getStringIndex(pos);

            if (found != unkPOSIdx) {
                priority.push_back(found);
            }
        }

        NP_priority.push_back(priority);
    }

    label2ChunkType[ADJP_STR] = ADJP_ID;
    label2ChunkType[ADVP_STR] = ADVP_ID;
    label2ChunkType[CONJP_STR] = CONJP_ID;
    label2ChunkType[INTJ_STR] = INTJ_ID;
    label2ChunkType[LST_STR]  = LST_ID;
    label2ChunkType[NP_STR] = NP_ID;
    label2ChunkType[O_STR] = O_ID;
    label2ChunkType[PP_STR] = PP_ID;
    label2ChunkType[PRT_STR] = PRT_ID;
    label2ChunkType[SBAR_STR] = SBAR_ID;
    label2ChunkType[UCP_STR] = UCP_ID;
    label2ChunkType[VP_STR] = VP_ID;

    labelIdx2ChunkType.resize(labelDict->size(), -1);
    const std::vector<std::string> &labels = labelDict->getKnownStringVector();
    for (const std::string &label : labels) {
        if (label[0] == 'E' || label[0] == 'S') {
            std::string labelType = label.substr(2, label.size() - 2);
            if (label2ChunkType.find(labelType) != label2ChunkType.end()) {
                int labelIdx = labelDict->getStringIndex(label);
                int labelTypeIdx = label2ChunkType[labelType];

                labelIdx2ChunkType[labelIdx] = labelTypeIdx;
            }
        } else if (label[0] == 'O') {
            int labelIdx = labelDict->getStringIndex(label);
            int labelTypeIdx = label2ChunkType[label];

            labelIdx2ChunkType[labelIdx] = labelTypeIdx;
        }
    }
}

int HeadWordRule::findHeadPosition(const std::vector<int> &POSs, const int start, const int end, const int labelIdx) {
    if (labelIdx2ChunkType[labelIdx] == -1) {
        std::cerr << "Wrong labelIdx: " << labelIdx << std::endl;
    }
    assert(labelIdx2ChunkType[labelIdx] != -1);

    int chunkTypeIdx = labelIdx2ChunkType[labelIdx];
    int headIndex = -1;
    switch (chunkTypeIdx) {
    case ADJP_ID:
        headIndex = findHeadInADJP(POSs, start, end);
        break;
    case ADVP_ID:
        headIndex = findHeadInADVP(POSs, start, end);
        break;
    case CONJP_ID:
        headIndex = findHeadInCONJP(POSs, start, end);
        break;
    case INTJ_ID:
        headIndex = start;
        break;
    case LST_ID:
        headIndex = findHeadInLST(POSs, start, end);
        break;
    case NP_ID:
        headIndex = findHeadInNP(POSs, start, end);
        break;
    case O_ID:
        headIndex = start;
        break;
    case PP_ID:
        headIndex = findHeadInPP(POSs, start, end);
        break;
    case PRT_ID:
        headIndex = findHeadInPRT(POSs, start, end);
        break;
    case SBAR_ID:
        headIndex = findHeadInSBAR(POSs, start, end);
        break;
    case UCP_ID:
        headIndex = end;
        break;
    case VP_ID:
        headIndex = findHeadInVP(POSs, start, end);
        break;
    default:
        std::cerr << "Label type wrong in findHeadPosition!" << std::endl;
        exit(0);
    }

    return headIndex;
}

int HeadWordRule::findHeadInADJP(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : ADJP_priority) {
        for (int j = start; j <= end; j++) {
            if (pos == POSs[j]) {
                return j;
            }
        }
    }

    return start;
}

int HeadWordRule::findHeadInADVP(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : ADVP_priority) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInCONJP(const std::vector<int> &POSs,const int start, const int end){
    for (int pos : CONJP_priority) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInLST(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : LST_priority) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInNP(const std::vector<int> &POSs,const int start, const int end) {
    if (NP_priority[0].size() > 0 && POSs[end] == NP_priority[0][0]) {
        return end;
    }
    for (int pos : NP_priority[1]) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }
    for (int pos : NP_priority[2]) {
        for (int j = start; j <= end; j++) {
            if (pos == POSs[j]) {
                return j;
            }
        }
    }
    for (int pos : NP_priority[3]) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }
    for (int pos : NP_priority[4]) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }
    for (int pos : NP_priority[5]) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInPP(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : PP_priority) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInPRT(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : PRT_priority) {
        for (int j = end; j >= start; j--) {
            if (pos == POSs[j])
                return j;
        }
    }

    return end;
}

int HeadWordRule::findHeadInSBAR(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : SBAR_priority) {
        for (int j = start; j <= end; j++) {
            if (pos == POSs[j]) {
                return j;
            }
        }
    }

    return start;
}

int HeadWordRule::findHeadInVP(const std::vector<int> &POSs,const int start, const int end) {
    for (int pos : VP_priority) {
        for (int j = start; j <= end; j++) {
            if (pos == POSs[j]) {
                return j;
            }
        }
    }

    return start;
}
