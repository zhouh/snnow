/*************************************************************************
	> File Name: parser.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/

#include <gflags/gflags.h>
#include <iostream>
#include <fstream>

#include "DepParseDataSet.h"
#include "DepParser.h"

/*
 * define the command line parameters
 */

DEFINE_string(embedding_file, "./data/sen.emb", "Embedding file for pre-training");
DEFINE_string(training_file, "./data/SD/small.train", "Traing file name");
DEFINE_string(test_file, "./data/SD/small.train", "Testing file name");
DEFINE_string(dev_file, "./data/SD/small.train", "Dev file name");
DEFINE_string(model_file, "./data/model.txt", "model file name");

DEFINE_int32(max_training_iteration_num, 2000, "The max number of training iterations to perform");
DEFINE_int32(batch_size, 10, "mini batch size");
DEFINE_int32(thread_num, 1, "thread num for multi-thread training");
DEFINE_int32(word_embedding_dim, 50, "The dimention of pre-trained word embedding");
DEFINE_int32(label_num, 28, "label num");
DEFINE_int32(beam_size, 16, "beam size for beam search");
DEFINE_int32(hidden_size, 200, "the hidden size of the neural network");
DEFINE_int32(feature_num, 48, "the total number of atomic features");
DEFINE_int32(evaluate_per_iteration, 10, "evaluation gap num between iterations");

DEFINE_bool(be_dropout, true, "whether using dropout");

DEFINE_double(learning_rate, 0.1, "learning rate");
DEFINE_double(init_range, 0.1, "the init range of the adaGrad updating");
DEFINE_double(regularization_rate, 1e-8, "rate for regularization");
DEFINE_double(dropout_prob, 0.5, "dropout probability");
DEFINE_double(adagrad_eps, 1e-6, "bias for the adaGrad updating");


int main(int argc, char* argv[]){

    google::ParseCommandLineFlags(&argc, &argv, true);

    std::cout<<"Begin to Training!"<<std::endl;

    //CConfig::ReadConfig( argv[1] );
    DepParser parser(true);

    // load data
    std::clog<< "### Begin to load data."<<std::endl;
    DepParseDataSet training_data(FLAGS_training_file);
    DepParseDataSet dev_data(FLAGS_dev_file);

    //output tree
    DepParseTree* parse_tree_ptr = static_cast<DepParseTree*>(training_data.outputs[0]);
    std::clog<<*parse_tree_ptr;
    std::clog<< "### End to load data."<<std::endl;

    // begin to train
    parser.train(static_cast<DataSet*>(&training_data), static_cast<DataSet*>(&dev_data));
}
