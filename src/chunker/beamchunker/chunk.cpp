/*************************************************************************
	> File Name: chunker.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 18 Nov 2015 02:15:36 PM CST
 ************************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Config.h"
#include "LabeledSequence.h"
#include "Instance.h"
#include "BeamChunker.h"

using std::cerr;
using std::endl;
using std::vector;

#define PRINT_CONFIGURATION

#define DEBUG


#ifdef DEBUG
//#define DEBUG1
#endif //! DEBUG

int main(int argc, char *argv[]) {
    std::cerr << "[BeamChunker Description]: ";
    if (argc == 1) {
        std::cerr << "There is no description!" << std::endl;
    } else {
        std::cerr << argv[1] << std::endl;
    }

#ifdef PRINT_CONFIGURATION
    std::cerr << "[begin]Configuration setting:" << std::endl;

    CConfig config;
    std::cerr << config;

    std::cerr << "[end]" << std::endl;
#endif

    std::ifstream isTrain(CConfig::strTrainPath);
    std::ifstream isDev(CConfig::strDevPath);

    ChunkedDataSet trainGoldSentences;
    InstanceSet trainInstances;

    ChunkedDataSet devGoldSentences;
    InstanceSet devInstances;

    int count = 0;
    while (true) {
        LabeledSequence cs;

        if (!(isTrain >> cs)) break;

        trainGoldSentences.push_back(cs);

        SequenceInput ci;
        cs.getSequenceInput(ci);
        Instance inst(ci);
        trainInstances.push_back(inst);

        count++;
    }

    cerr << "Training sentences number: " << count << endl;

    count = 0;
    while (true) {
        LabeledSequence cs;

        if (!(isDev >> cs)) {
            break;
        }

        devGoldSentences.push_back(cs);

        SequenceInput ci;
        cs.getSequenceInput(ci);
        Instance inst(ci);
        devInstances.push_back(inst);

        count++;
    }

    BeamChunker chunker(true);
    cerr << "Dev sentences number: " << count << endl;
    cerr << "------------------------------------" << endl;

    cerr << "Chunker: start training..." << endl;
    chunker.train(trainGoldSentences, trainInstances, devGoldSentences, devInstances);

    return 0;
}
