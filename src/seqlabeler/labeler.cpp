/*************************************************************************
	> File Name: labeler.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:33:12 PM CST
 ************************************************************************/
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>

#include "SeqLabeler.h"
#include "SeqLabelerDataSet.h"

DEFINE_string(embedding_file, "./data/chunk/sen.emb", "Embedding file for pre-training");
DEFINE_string(dict_file, "./data/chunk/giga.dict", "Dictionary file of the most 100,000 frequent words appearing in giga dataset.");
DEFINE_string(test_dict_file, "./data/chunk/test.dict", "Dictionary file of the intersection of words appearing in dict_file and test dataset.");
DEFINE_string(training_file, "./data/chunk/small.train", "Traing file name");
DEFINE_string(test_file, "./data/chunk/small.train", "Testing file name");
DEFINE_string(dev_file, "./data/chunk/small.train", "Dev file name");
DEFINE_string(model_file, "./data/chunk/model.txt", "model file name");

DEFINE_int32(max_training_iteration_num, 1000, "The max number of training iterations to perform");
DEFINE_int32(batch_size, 100, "mini batch size");
DEFINE_int32(thread_num, 1, "thread num for multi-thread training");
DEFINE_int32(word_embedding_dim, 50, "The dimention of pre-trained word embedding");
DEFINE_int32(label_num, 28, "label num");
DEFINE_int32(beam_size, 16, "beam size for beam search");
DEFINE_int32(hidden_size, 20, "the hidden size of the neural network");
DEFINE_int32(feature_num, 48, "the total number of atomic features");
DEFINE_int32(evaluate_per_iteration, 10, "evaluation gap num between iterations");

DEFINE_bool(be_dropout, true, "whether using dropout");

DEFINE_double(learning_rate, 0.05, "learning rate");
DEFINE_double(init_range, 0.1, "the init range of the adaGrad updating");
DEFINE_double(regularization_rate, 1e-8, "rate for regularization");
DEFINE_double(dropout_prob, 0.5, "dropout probability");
DEFINE_double(adagrad_eps, 1e-6, "bias for the adaGrad updating");

int main(int argc, char* argv[]){
    std::clog << "Begin to train..." <<std::endl;

    //CConfig::ReadConfig( argv[1] );
    SeqLabeler labeler(true);

    // load data
    std::clog << "Begin to load data..." << std::endl;
    SeqLabelerDataSet training_data(FLAGS_training_file);
    // std::cout << training_data << std::endl;
    SeqLabelerDataSet dev_data(FLAGS_training_file);

    // //output tree
    // DepParseTree* parse_tree_ptr = static_cast<DepParseTree*>(training_data.outputs[0]);
    // std::clog<<*parse_tree_ptr;
    std::clog << "End to load data." << std::endl;

    // begin to train
    labeler.train(static_cast<DataSet&>(training_data), static_cast<DataSet&>(dev_data));
}
