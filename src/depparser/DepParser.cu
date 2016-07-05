//
// Created by zhouh on 16-4-5.
//
#include <ctime>
#include <memory>

#include "DepParser.h"

using namespace mshadow;
using namespace mshadow::expr;

void printVector(std::vector<int> vec) {

    for (int i = 0; i < vec.size(); i++) {
        std::cout << "<" << i << ">=" << vec[i] << std::endl;
    }

}

/**
 * initialize the feature extractor and
 * transition system handlers for parser
 */
DepParser::DepParser(bool bTrain) {
    beam_size = FLAGS_beam_size;
    be_train = bTrain;
    trainsition_system_ptr.reset(new DepArcStandardSystem());
    feature_extractor_ptr.reset(new DepParseFeatureExtractor());
}

/**
 *  do the training init for formal training
 *
 *  0. init the feature type of this system
 *  1. get dictionary for the feature extractor
 *  2. init the transition system handler for the parser
 *  3. init the feature embedding handler
 */
void DepParser::trainInit(DataSet *training_data_ptr) {

    std::clog << "======================================" << std::endl;
    std::clog << "Training Init!" << std::endl;
    std::clog << "Training Instance Num: " << training_data_ptr->getSize() << std::endl;
    std::clog << "======================================" << std::endl;


    std::clog << "###Begin to init the feature types of this system: " << std::endl;


    // prepare the handler for parsing
    std::clog << "###Begin to init the dictionaries!" << std::endl;
    feature_extractor_ptr->getDictionaries(training_data_ptr);  // dictionary for feature index
    feature_extractor_ptr->displayDict();
    std::clog << "###End to init the dictionaries!" << std::endl;

    std::clog << "###Begin to create feature types!" << std::endl;

    /*
     * total 3 feature types for dependency parsing
     * word feature
     * tag feature
     * label feature
     */
    FeatureTypes feature_types;
    FeatureType word_feat_type(DepParseFeatureExtractor::word_string,
                               feature_extractor_ptr->feature_nums[DepParseFeatureExtractor::c_word_dict_index],
                               feature_extractor_ptr->dictionary_ptrs_table[DepParseFeatureExtractor::c_word_dict_index]->size(),
                               c_word_feature_dim);
    FeatureType tag_feat_type(DepParseFeatureExtractor::tag_string,
                              feature_extractor_ptr->feature_nums[DepParseFeatureExtractor::c_tag_dict_index],
                              feature_extractor_ptr->dictionary_ptrs_table[DepParseFeatureExtractor::c_tag_dict_index]->size(),
                              c_tag_feature_dim);
    FeatureType label_feat_type(DepParseFeatureExtractor::label_string,
                                feature_extractor_ptr->feature_nums[DepParseFeatureExtractor::c_dep_label_dict_index],
                                feature_extractor_ptr->dictionary_ptrs_table[DepParseFeatureExtractor::c_dep_label_dict_index]->size(),
                                c_label_feature_dim);
    feature_types.push_back(word_feat_type);
    feature_types.push_back(tag_feat_type);
    feature_types.push_back(label_feat_type);

    // set the feature types for feature handlers
    feature_extractor_ptr->setFeatureTypes(feature_types);
    FeatureVector::setFeatureTypes(feature_types);
    std::clog << "###End to create feature types!" << std::endl;


    // init transition system
    std::clog << "###Init the transition system!" << std::endl;
    trainsition_system_ptr->makeTransition(feature_extractor_ptr->getKnownDepLabelVector(),
                                           feature_extractor_ptr->getKnownDepLabelVectorMap());

    std::clog << "###Begin to generate the training examples!" << std::endl;

    feature_extractor_ptr->generateGreedyTrainingExamples(trainsition_system_ptr.get(),
                                                          static_cast<DepParseDataSet *>(training_data_ptr),
                                                          greedy_example_ptrs);
    std::clog << "Constructing dictionary and training examples done!" << std::endl;
}

void DepParser::train(DataSet *train_data_ptr, DataSet *dev_data_ptr) {

    // init training
    trainInit(train_data_ptr);

    /*
     * prepare for the neural networks, every parsing step maintains a specific net
     * because each parsing step has different updating gradients.
     */
    const int num_in = feature_extractor_ptr->getTotalInputSize();
    const int num_hidden = FLAGS_hidden_size;
    const int num_out = trainsition_system_ptr->action_factory_ptr->total_action_num;
//    const int beam_size = FLAGS_beam_size;
    const int batch_size = std::min(FLAGS_batch_size, static_cast<int>(greedy_example_ptrs.size()));
//    const bool be_dropout = FLAGS_dropout_prob;


    /*
     * create the model for training
     */
    std::clog << "###Begin to construct training model." << std::endl;
    Stream <gpu> *stream = NewStream<gpu>();
    Model<gpu> model(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, stream);
    model.randomInitialize();
    Model<gpu> adagrad_squares(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types,
                               stream);  // for adagrad updating
    std::clog << "###End to construct training model." << std::endl;


    double best_uas = -1;
    for (int iter = 1; iter <= FLAGS_max_training_iteration_num; iter++) {

        std::clog << "iteration\t" << iter << std::endl;

        // record the cost time
        auto start = std::chrono::high_resolution_clock::now();

        // random shuffle the training instances in the container,
        // get the shuffled training data for the mini-batch training of this iteration
        std::cout << "shuffle the training data." << std::endl;
        std::random_shuffle(greedy_example_ptrs.begin(), greedy_example_ptrs.end());
        int batch_example_index_end = std::min(batch_size, static_cast<int>(greedy_example_ptrs.size()));
        std::vector<std::shared_ptr<Example>> multiThread_miniBatch_data(greedy_example_ptrs.begin(),
                                                                         greedy_example_ptrs.begin() +
                                                                         batch_example_index_end);

        // cumulated gradients for updating
//        Model<gpu> batch_cumulated_grads(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, NULL);
        Model<gpu> gradients(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, stream);

        // create the neural net for prediction
        std::shared_ptr<FeedForwardNNet<gpu>> nnet;
        nnet.reset(new FeedForwardNNet<gpu>(batch_size, num_in, num_hidden, num_out, &model));

        // feature vector lists for action sequence
        FeatureVectors feature_vectors(multiThread_miniBatch_data.size());

        // batch input of
        TensorContainer<cpu, 2, real_t> input(Shape2(batch_size, num_in));

        std::vector<std::vector<int>> valid_action_vectors(multiThread_miniBatch_data.size());

        TensorContainer<cpu, 2, real_t> batch_predict_output(Shape2(batch_size, num_out));

        /*
         * init the input and predict output
         */
        input = 0.0;
        batch_predict_output = 0.0;

        // fill the feature vectors for batch training


        // prepare batch training data!
        for (int inst = 0; inst < multiThread_miniBatch_data.size(); inst++) {
            auto e = greedy_example_ptrs[inst];

            feature_vectors[inst] = e->feature_vector;
            valid_action_vectors[inst] = e->predict_labels;
        }

        feature_extractor_ptr->returnInput(feature_vectors, model.featEmbs, input);

//        for (int j = 0; j < input.size(0); ++j) {
//            for (int i = 0; i < input.size(1); ++i) {
//                std::cout << input[j][i] << "\t";
//            }
//            std::cout << std::endl;
//
//        }

        nnet->Forward(input, batch_predict_output, FLAGS_be_dropout);


        int total_correct_predict_action_num = 0;
        double loss = 0;
        for (int inst = 0; inst < multiThread_miniBatch_data.size(); inst++) {

            int opt_act = -1;
            int gold_act = -1;

            std::vector<int> &valid_acts = valid_action_vectors[inst];

            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {

                    std::cout << "batch_predict_output\t"<<i<<"\t"<<batch_predict_output[inst][i] << std::endl;
                    if (opt_act == -1 || batch_predict_output[inst][i] > batch_predict_output[inst][opt_act]) {
                        opt_act = i;
                    }

                    if (valid_acts[i] == 1) {
                        gold_act = i;
                    }
                }
            }

            std::cout << inst <<": opt\t"<<opt_act<< std::endl;
            std::cout << inst <<": gold\t"<<gold_act<< std::endl;

            if (opt_act == gold_act) {
                total_correct_predict_action_num += 1;
            }

            real_t max_score = batch_predict_output[inst][opt_act];
            real_t gold_score = batch_predict_output[inst][gold_act];

//            real_t sum = 0.0;
//
//            for (int i = 0; i < valid_acts.size(); i++) {
//                if (valid_acts[i] >= 0) {
//                    batch_predict_output[inst][i] = std::exp(batch_predict_output[inst][i] - max_score);
//                    sum += batch_predict_output[inst][i];
//                }
//            }
//
//            loss += (std::log(sum) - (gold_score - max_score)) / multiThread_miniBatch_data.size();
//
//            for (int i = 0; i < valid_acts.size(); i++) {
//                if (valid_acts[i] >= 0) {
//                    batch_predict_output[inst][i] = batch_predict_output[inst][i] / sum;
//                } else {
//                    batch_predict_output[inst][i] = 0.0;
//                }
//            }
//            batch_predict_output[inst][gold_act] -= 1.0;
//        }

            const int act_num = valid_acts.size();
            std::vector<real_t> lx(act_num, 0.0);
            std::vector<real_t> x_bar(act_num, 0.0);
            std::vector<real_t> x_barExp(act_num, 0.0);
            std::vector<real_t> y_bar(act_num, 0.0);
            std::vector<real_t> z_bar(act_num, 0.0);
            real_t tloss = 0.0;
            real_t Q = 0.0;
            for (int i = 0; i < act_num; i++) {
                if (valid_acts[i] >= 0) {
                    x_bar[i] = batch_predict_output[inst][i] - max_score;
                    x_barExp[i] = exp(x_bar[i]);
                    Q += x_barExp[i];
                }
            }
            for (int i = 0; i < act_num; i++) {
                if (valid_acts[i] >= 0) {
                    y_bar[i] = x_barExp[i] / Q;
                }
            }
            for (int i = 0; i < act_num; i++) {
                if (valid_acts[i] >= 0) {
                    if (y_bar[i] <= 0.5) {
                        z_bar[i] = 1.0 - y_bar[i];
                    } else {
                        z_bar[i] = (Q - x_barExp[i]) / Q;
                    }
                }
            }
            for (int i = 0; i < act_num; i++) {
                real_t tbar = (i == gold_act) ? 1.0 : 0.0;

                if (valid_acts[i] >= 0) {
                    if (y_bar[i] <= 0.5) {
                        lx[i] = y_bar[i] - tbar;

                        tloss += tbar * std::log(y_bar[i]);
                    } else {
                        lx[i] = (1.0 - tbar) - z_bar[i];

                        tloss += tbar * std::log(1.0 - z_bar[i]);
                    }
                }
            }

            loss -= tloss;

            for (int i = 0; i < act_num; i++) {
                batch_predict_output[inst][i] = lx[i];
            }
        }

//        batch_predict_output /= batch_size;

        batch_predict_output /= static_cast<real_t>(multiThread_miniBatch_data.size());

        nnet->Backprop(batch_predict_output);
        nnet->SubsideGradsTo(&gradients, feature_vectors); // add the gradients from the nets to the gradients
        model.update(&gradients,
                     &adagrad_squares); // update the gradient with adagrad, and update the parameters in the models
        auto end = std::chrono::high_resolution_clock::now();

        if (iter % FLAGS_evaluate_per_iteration == 0) {
            double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
            std::clog << "[" << iter << "] totally train " << batch_size << " examples, time: " << time_used <<
            " average: " << batch_size / time_used << " examples/second!" << std::endl;

            double posClassificationRate = static_cast<double>(total_correct_predict_action_num) / batch_size;
            double regular_loss = 0.5 * FLAGS_regularization_rate * model.norm2();
            double avg_loss = (loss + regular_loss) / batch_size;
            std::clog << "current objective fun-score  : " << avg_loss << "\tclassfication rate: " <<
            posClassificationRate <<

            /*
         * do the evaluation in iteration of training
         * save the best resulting model
         */
            std::cout << "###Test Begin###" << std::endl;
            // do the evaluation
            double dev_uas = test(dev_data_ptr, &model, nnet.get());
            std::clog << "Current Iteration UAS\t" << dev_uas << "%" << std::endl;
            if (dev_uas > best_uas) {
                std::ofstream ofs(FLAGS_model_file);
                model.saveModel(ofs);
                ofs.close();

            }
        }

    }


}

//===============================================================================
double DepParser::test(DataSet *test_data, Model<gpu> *model, FeedForwardNNet<gpu> *net) {

    const int num_in = feature_extractor_ptr->getTotalInputSize();
//    const int num_hidden = FLAGS_hidden_size;
    const int num_out = trainsition_system_ptr->action_factory_ptr->total_action_num;

    std::vector<DepParseTree> predict_trees(test_data->size);

    std::vector<DepParseTree> gold_dep_trees;

    std::shared_ptr<FeedForwardNNet<gpu>> nnet;
    nnet.reset(new FeedForwardNNet<gpu>(1, num_in, FLAGS_hidden_size, num_out, model));

    for (int inst = 0; inst < test_data->size; ++inst) {

        auto &tree_i = static_cast<DepParseTree & >(*(test_data->outputs[inst]));
        auto &input_i = static_cast<DepParseInput & >(*(test_data->inputs[inst]));

        feature_extractor_ptr->getCache(input_i);

        gold_dep_trees.push_back(tree_i);



        // n shift and n reduce, one more reduce action for root
        int total_act_num_one_sentence = (input_i.size() - 1) * 2;

        /*
         * cache the dependency label in the training set
         */
        std::vector<int> labelIndexCache(tree_i.size);
        int index = 0;
        for (auto iter = tree_i.nodes.begin(); iter != tree_i.nodes.end();
             iter++) {
            int labelIndex = feature_extractor_ptr->getLabelIndex(iter->label);

            if (labelIndex == -1) {
                std::cerr << "Dep label " << iter->label
                << " is not in labelMap!" << std::endl;
                exit(1);
            }

            labelIndexCache[index] = labelIndex;
            index++;
        }

        std::shared_ptr<DepParseState> state_ptr;
        state_ptr.reset(new DepParseState());

        state_ptr->len_ = input_i.size();
        state_ptr->initCache();
//        getCache(input_i);

//        state_ptr->toString();
        //for every state of a sentence
        for (int j = 0; !state_ptr->complete(); j++) {

            TensorContainer<cpu, 2, real_t> input(Shape2(1, num_in));
            TensorContainer<cpu, 2, real_t> batch_predict_output(Shape2(1, num_out));

            /*
             * init the input and predict output
             */
            input = 0.0;
            batch_predict_output = 0.0;

            std::vector<int> valid_acts;

            //get current state features
            FeatureVector fv = feature_extractor_ptr->getFeatureVectors(
                    static_cast<State *>(state_ptr.get()),
                    static_cast<Input *>(&input_i));
            FeatureVectors fvs;
            fvs.push_back(fv);

            //get current state valid actions
            trainsition_system_ptr->getValidActs(static_cast<State *>(state_ptr.get()), valid_acts);

//            printVector(valid_acts);

            feature_extractor_ptr->returnInput(fvs, model->featEmbs, input);

            nnet->Forward(input, batch_predict_output, FLAGS_dropout_prob);


            int opt_act = -1;
            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    if (opt_act == -1 || batch_predict_output[inst][i] > batch_predict_output[inst][opt_act]) {
                        opt_act = i;
                    }
                }
            }

            trainsition_system_ptr->Move(static_cast<State *>(state_ptr.get()),
                                         DepParseShiftReduceActionFactory::action_table[opt_act].get());
//            std::cout<<"action:\t"<<static_cast<DepParseAction*>(DepParseShiftReduceActionFactory::action_table[opt_act].get())->toString(feature_extractor_ptr->dictionary_ptrs_table[feature_extractor_ptr->c_dep_label_dict_index])<<std::endl;
//            std::cout<<"label code\t"<<DepParseShiftReduceActionFactory::action_table[opt_act]->getActionLabel()<<std::endl;

//            state_ptr->toString();
        }


        // generate the predict tree from the complete state
        trainsition_system_ptr->GenerateOutput(static_cast<State *>(state_ptr.get()),
                                               static_cast<Input *>(&input_i),
                                               static_cast<Output *>( &(predict_trees[inst])));

        std::cout << "output tree:" << std::endl;
        std::cout << predict_trees[inst] << std::endl;


    }

    DepParseEvalb evalb;
    double result = evalb.evalb(predict_trees, gold_dep_trees);
    return result * 100;
}