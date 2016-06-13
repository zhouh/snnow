//
// Created by zhouh on 16-4-5.
//
#include <ctime>
#include <memory>

#include "DepParser.h"

using namespace mshadow;
using namespace mshadow::expr;

/**
 * initialize the feature extractor and
 * transition system handlers for parser
 */
DepParser::DepParser(bool bTrain){
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
void DepParser::trainInit(DataSet &training_data) {

    std::clog << "======================================";
    std::clog << "Training Init!" << std::endl;
    std::clog << "Training Instance Num: " << training_data.getSize() << std::endl;
    std::clog << "======================================";


    std::clog << "###Begin to init the feature types of this system: " << std::endl;


    // prepare the handler for parsing
    std::clog << "###Begin to init the dictionaries!" << std::endl;
    feature_extractor_ptr->getDictionaries(training_data);  // dictionary for feature index
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

    feature_extractor_ptr->generateGreedyTrainingExamples(trainsition_system_ptr.get(), static_cast<DepParseDataSet&>(training_data), greedy_example_ptrs);
    std::clog << "Constructing dictionary and training examples done!" << std::endl;
}

void DepParser::train(DataSet &train_data, DataSet &dev_data) {

    // init training
    trainInit(train_data);

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
    Model<cpu> model(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, NULL);
    Model<cpu> adagrad_squares(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types,
                               NULL);  // for adagrad updating
    Stream <gpu> *stream = stream = NewStream<gpu>();
    std::clog << "###End to construct training model." << std::endl;


    double best_uas = -1;
    for (int iter = 1; iter <= FLAGS_max_training_iteration_num; iter++) {

        // record the cost time
        auto start = std::chrono::high_resolution_clock::now();

        // random shuffle the training instances in the container,
        // get the shuffled training data for the mini-batch training of this iteration
        std::random_shuffle(greedy_example_ptrs.begin(), greedy_example_ptrs.end());
        int batch_example_index_end = std::min(batch_size, static_cast<int>(greedy_example_ptrs.size()));
        std::vector<std::shared_ptr<Example>> multiThread_miniBatch_data(greedy_example_ptrs.begin(),
        greedy_example_ptrs.begin() + batch_example_index_end);

        // cumulated gradients for updating
        Model<cpu> batch_cumulated_grads(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, NULL);
        Model<gpu> gradients(num_in, num_hidden, num_out, feature_extractor_ptr->feature_types, stream);

        // create the neural net for prediction
        std::shared_ptr<FeedForwardNNet<gpu>> nnet;
        nnet.reset(new FeedForwardNNet<gpu>(batch_size, num_in, num_hidden, num_out, &model));

        // feature vector lists for action sequence
        FeatureVectors feature_vectors(multiThread_miniBatch_data.size());

        // batch input of
        TensorContainer<cpu, 2, real_t> input(Shape2(batch_size, num_in));

        std::vector<std::vector<int>> valid_action_vectors(multiThread_miniBatch_data.size(), std::vector<int>(num_out,0));

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

        nnet->Forward(input, batch_predict_output, FLAGS_dropout_prob);


        int total_correct_predict_action_num = 0;
        double loss = 0;
        for (int inst = 0; inst < multiThread_miniBatch_data.size(); inst++) {

            int opt_act = -1;
            int gold_act = -1;

            std::vector<int> &valid_acts = valid_action_vectors[inst];

            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    if (opt_act == -1 || batch_predict_output[inst][i] > batch_predict_output[inst][opt_act]) {
                        opt_act = i;
                    }

                    if (valid_acts[i] == 1) {
                        gold_act = i;
                    }
                }
            }

            if (opt_act == gold_act) {
                total_correct_predict_action_num += 1;
            }

            real_t max_score = batch_predict_output[inst][opt_act];
            real_t gold_score = batch_predict_output[inst][gold_act];

            real_t sum = 0.0;

            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    batch_predict_output[inst][i] = std::exp(batch_predict_output[inst][i] - max_score);
                    sum += batch_predict_output[inst][i];
                }
            }

            loss += (std::log(sum) - (gold_score - max_score)) / multiThread_miniBatch_data.size();

            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    batch_predict_output[inst][i] = batch_predict_output[inst][i] / sum;
                } else {
                    batch_predict_output[inst][i] = 0.0;
                }
            }
            batch_predict_output[inst][gold_act] -= 1.0;
        }

        batch_predict_output /= static_cast<real_t>(multiThread_miniBatch_data.size());

        nnet->Backprop(batch_predict_output);
        nnet->SubsideGradsTo(&gradients, feature_vectors);

        model.update(&batch_cumulated_grads, &adagrad_squares);
        auto end = std::chrono::high_resolution_clock::now();

        if (iter % FLAGS_evaluate_per_iteration == 0) {
            double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 ;
            std::clog << "[" << iter << "] totally train " << batch_size << " examples, time: " << time_used <<
            " average: " << batch_size / time_used << " examples/second!" << std::endl;
        }

        /*
         * do the evaluation in iteration of training
         * save the best resulting model
         */
        if (iter % FLAGS_evaluate_per_iteration == 0) {
            // do the evaluation
            double dev_uas = test(dev_data, model, nnet.operator*());
            if (dev_uas > best_uas){
                std::ofstream ofs(FLAGS_model_file);
                model.saveModel(ofs);
                ofs.close();

            }
        }
    }


}

//===============================================================================
double DepParser::test(DataSet &test_data, Model<cpu> & model, FeedForwardNNet<gpu> & net) {

    const int num_in = feature_extractor_ptr->getTotalInputSize();
//    const int num_hidden = FLAGS_hidden_size;
    const int num_out = trainsition_system_ptr->action_factory_ptr->total_action_num;

    std::vector<DepParseTree> predict_trees(test_data.size);

    std::vector<DepParseTree> gold_dep_trees;

    for (int inst = 0; inst < test_data.size; ++inst) {

        auto & tree_i = static_cast<DepParseTree& >(test_data.outputs[inst]);
        auto & input_i = static_cast<DepParseInput& >(test_data.inputs[inst]);

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

        //for every state of a sentence
        for (int j = 0; !state_ptr->complete(); j++) {

            TensorContainer<cpu, 2, real_t> input(Shape2(1, num_in));
            TensorContainer<cpu, 2, real_t> batch_predict_output(Shape2(1, num_out));

            /*
             * init the input and predict output
             */
            input = 0.0;
            batch_predict_output = 0.0;

            std::vector<int> valid_acts(total_act_num_one_sentence, 0);

            //get current state features
            FeatureVector fv = feature_extractor_ptr->getFeatureVectors(*state_ptr, input_i);
            FeatureVectors fvs;
            fvs.push_back(fv);

            //get current state valid actions
            trainsition_system_ptr->getValidActs(state_ptr.operator*(), valid_acts);

            feature_extractor_ptr->returnInput(fvs, model.featEmbs, input);

            net.Forward(input, batch_predict_output, FLAGS_dropout_prob);


            int opt_act = -1;
            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    if (opt_act == -1 || batch_predict_output[inst][i] > batch_predict_output[inst][opt_act]) {
                        opt_act = i;
                    }
                }
            }

            trainsition_system_ptr->Move(*state_ptr, DepParseShiftReduceActionFactory::action_table[opt_act].operator*());
        }


        // generate the predict tree from the complete state
        trainsition_system_ptr->GenerateOutput( *state_ptr, input_i, predict_trees[inst] );
    }

    DepParseEvalb evalb;
    double result = evalb.evalb(predict_trees, gold_dep_trees);
    return result * 100;
}