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
#include "ChunkedSentence.h"
#include "Instance.h"
#include "GreedyChunker.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

#define DEBUG


#ifdef DEBUG
//#define DEBUG1
#endif //! DEBUG

int main(int argc, char *argv[]) {
    std::cerr << "Chunker: init..." << endl;

    std::ifstream isTrain(CConfig::strTrainPath);
    std::ifstream isDev(CConfig::strDevPath);

    ChunkedDataSet trainGoldSentences;
    InstanceSet trainInstances;

    ChunkedDataSet devGoldSentences;
    InstanceSet devInstances;

    int count = 0;
    while (true) {
        ChunkedSentence cs;

        if (!(isTrain >> cs)) break;

#ifdef DEBUG1
        cout << cs << endl;
        cout << "size: " << cs.size() << endl;
#endif //!DEBUG1

        trainGoldSentences.push_back(cs);

        ChunkerInput ci;
        cs.getChunkerInput(ci);
        Instance inst(ci);
        trainInstances.push_back(inst);

#ifdef DEBUG1
        inst.print();
#endif //! DEBUG1

        count++;
    }

    count = 0;
    while (true) {
        ChunkedSentence cs;

        if (!(isDev >> cs)) {
            break;
        }

#ifdef DEBUG1
        cout << cs << endl;
        cout << "size: " << cs.size() << endl;
#endif
        
        devGoldSentences.push_back(cs);

        ChunkerInput ci;
        cs.getChunkerInput(ci);
        Instance inst(ci);
        devInstances.push_back(inst);

#ifdef DEBUG1
        inst.print();
#endif

        count++;
    }

    GreedyChunker chunker(true);
    cerr << "Training sentence number: " << count << endl;
    cerr << endl;

    cerr << "Chunker: training..." << endl;

    chunker.train(trainGoldSentences, trainInstances, devInstances);

    return 0;
}
