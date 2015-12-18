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

#include "mshadow/tensor.h"
#include "Config.h"
#include "ChunkedSentence.h"
#include "Instance.h"
#include "GreedyChunker.h"
// #include "Chunker.h"

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

    cerr << "Training sentences number: " << count << endl;

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
    cerr << "Dev sentences number: " << count << endl;
    cerr << "------------------------------------" << endl;

    cerr << "Chunker: start training..." << endl;
    chunker.train(trainGoldSentences, trainInstances, devGoldSentences, devInstances);

    using namespace mshadow;
    using namespace mshadow::expr;

    // InitTensorEngine<cpu>();
    // InitTensorEngine<gpu>();

    // Stream<cpu> *stream = NewStream<cpu>();

    // Tensor<cpu, 2> ts1 = NewTensor<cpu, float>(Shape2(2, 2), 0.0f);

    // ts1[0][0] = 1;
    // ts1[0][1] = 2;
    // ts1[1][0] = 3;
    // ts1[1][1] = 4;

    // Tensor<gpu, 2> tg = NewTensor<gpu, float>(ts1.shape_, 0.0f);
    // Copy(tg, ts1);

    // for (int r = 0; r < tg.shape_[0]; r++) {
    //     for (int c = 0; c < tg.shape_[1]; c++) {
    //         cout << tg[r][c] << " ";
    //     }
    //     cout << endl;
    // }

    // ShutdownTensorEngine<cpu>();
    // ShutdownTensorEngine<gpu>();

    return 0;
}
