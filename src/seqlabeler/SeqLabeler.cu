/*************************************************************************
	> File Name: SeqLabeler.cu
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:33:34 PM CST
 ************************************************************************/
#include <chrono>

#include "SeqLabeler.h"

SeqLabeler::SeqLabeler(bool b_train) {
    b_train_ = b_train;

    feature_extractor_ptr_.reset(new ChunkerFeatureExtractor());
    transition_system_ptr_.reset(new ChunkerTransitionSystem());
}

void SeqLabeler::train(DataSet &training_set, DataSet &dev_set) {
    trainInit(&training_set);

    const int num_in = feature_extractor_ptr_->getTotalInputSize();
    const int num_hidden = FLAGS_hidden_size;
    const int num_out = transition_system_ptr_->action_factory_ptr_->total_action_num;
    const int batch_size = std::min(FLAGS_batch_size, static_cast<int>(greedy_example_ptrs_.size()));

    std::clog << "# Begin to construct training model..." << std::endl;
    Stream<gpu> *stream = NewStream<gpu>();
    Model<gpu> model(num_in, num_hidden, num_out, feature_extractor_ptr_->feature_types_, stream);
    model.randomInitialize();
    // Model<gpu>::readWordPreTrain(FLAGS_embedding_file, feature_extractor_ptr_->getWordDict(), model.featEmbs[feature_extractor_ptr_->c_word_dict_index_]);
    Model<gpu> adagrad_squares(num_in, num_hidden, num_out, feature_extractor_ptr_->feature_types_, stream);  // for adagrad updating
    std::clog << "# End to construct training model!" << std::endl;

    double best_fscore = -1;
    for (int iter = 1; iter <= FLAGS_max_training_iteration_num; iter++) {

        // record the cost time
        auto start = std::chrono::high_resolution_clock::now();

        // random shuffle the training instances in the container,
        // get the shuffled training data for the mini-batch training of this iteration
        std::random_shuffle(greedy_example_ptrs_.begin(), greedy_example_ptrs_.end());
        std::vector<std::shared_ptr<Example>> multiThread_miniBatch_data(greedy_example_ptrs_.begin(),
                                                                         greedy_example_ptrs_.begin() + batch_size);

        // cumulated gradients for updating
//        Model<gpu> batch_cumulateddddd_grads(num_in, num_hidden, num_out, feature_extractor_ptr_->feature_types_, stream);
        Model<gpu> gradients(num_in, num_hidden, num_out, feature_extractor_ptr_->feature_types_, stream);

        // // create the neural net for prediction
        std::shared_ptr<FeedForwardNNet<gpu>> nnet;
        nnet.reset(new FeedForwardNNet<gpu>(batch_size, num_in, num_hidden, num_out, &model));

        // // feature vector lists for action sequence
        FeatureVectors feature_vectors(batch_size);

        // // batch input of
        TensorContainer<cpu, 2, real_t> input(Shape2(batch_size, num_in));

        std::vector<std::vector<int>> valid_action_vectors(batch_size, std::vector<int>(num_out,0));

        TensorContainer<cpu, 2, real_t> batch_predict_output(Shape2(batch_size, num_out));

        // /*
        //  * init the input and predict output
        //  */
        input = 0.0;
        batch_predict_output = 0.0;

        // // fill the feature vectors for batch training

        // // prepare batch training data!
        for (int inst = 0; inst < batch_size; inst++) {
            auto e = greedy_example_ptrs_[inst];

            feature_vectors[inst] = e->feature_vector;
            valid_action_vectors[inst] = e->predict_labels;
        }

        feature_extractor_ptr_->returnInput(feature_vectors, model.featEmbs, input);

        nnet->ChunkForward(input, batch_predict_output, FLAGS_be_dropout);

        int total_correct_predict_action_num = 0;
        double loss = 0;
        for (int insti = 0; insti < batch_size; insti++) {
            int opt_act = -1;
            int gold_act = -1;

            std::vector<int> &valid_acts = valid_action_vectors[insti];

            for (int i = 0; i < valid_acts.size(); i++) {
                if (valid_acts[i] >= 0) {
                    if (opt_act == -1 || batch_predict_output[insti][i] > batch_predict_output[insti][opt_act]) {
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

            real_t max_score = batch_predict_output[insti][opt_act];
            real_t gold_score = batch_predict_output[insti][gold_act];

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
                    x_bar[i] = batch_predict_output[insti][i] - max_score;
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
                        lx[i] = ( 1.0 - tbar ) - z_bar[i];

                        tloss += tbar * std::log( 1.0 - z_bar[i] );
                    }
                }
            }

            loss -= tloss;

            for (int i = 0; i < act_num; i++) {
                batch_predict_output[insti][i] = lx[i];
            }
        }

        batch_predict_output /= batch_size;

        nnet->ChunkBackprop(batch_predict_output);
        nnet->SubsideGradsTo(&gradients, feature_vectors);

        model.update(&g, &adagrad_squares);
        auto end = std::chrono::high_resolution_clock::now();

        if (iter % FLAGS_evaluate_per_iteration == 0) {
            double time_used = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 ;
            std::clog << "[" << iter << "] totally train " << batch_size << " examples, time: " << time_used <<
            " average: " << batch_size / time_used << " examples/second!" << std::endl;
            double posClassificationRate = static_cast<double>(total_correct_predict_action_num) / batch_size;
            double regular_loss = 0.5 * FLAGS_regularization_rate * model.norm2();
            double avg_loss = (loss  + regular_loss) / batch_size;
            std::cerr << "current objective fun-score  : " << avg_loss << "\tclassfication rate: " << posClassificationRate << std::endl;
        }

        /*
         * do the evaluation in iteration of training
         * save the best resulting model
         */
        // if (iter % FLAGS_evaluate_per_iteration == 0) {
        //     // do the evaluation
        //     double dev_uas = test(dev_data, model, nnet.operator*());
        //     if (dev_uas > best_fscore){
        //         std::ofstream ofs(FLAGS_model_file);
        //         model.saveModel(ofs);
        //         ofs.close();

        //     }
        // }
    }
}

void SeqLabeler::greedyTrain(DataSet &training_set, DataSet &dev_set) {
}

double SeqLabeler::test(DataSet &test_data, Model<cpu> &model, FeedForwardNNet <gpu> &net) {

    return 0.0;
}

void SeqLabeler::trainInit(DataSet *training_set_ptr) {
    std::clog << "======================================"<<std::endl;
    std::clog << "Training Init!" << std::endl;
    std::clog << "Training Instance Num: " << training_set_ptr->getSize() << std::endl;
    std::clog << "======================================"<<std::endl;

    // prepare the handler for parsing
    std::clog << "## Begin to init the dictionaries..." << std::endl;
    feature_extractor_ptr_->getDictionaries(training_set_ptr);  // dictionary for feature index
    feature_extractor_ptr_->displayDict();

    std::clog << "## End to init the dictionaries!" << std::endl;

    std::clog << "## Begin to create feature types..." << std::endl;
    feature_extractor_ptr_->setFeatureTypes();
    feature_extractor_ptr_->displayFeatureTypes();
    std::clog << "## End to create feature types!" << std::endl;

    std::clog << "## Begin to init the transition system..." << std::endl;
    transition_system_ptr_->makeTransitions(feature_extractor_ptr_->getKnownLabelVector(), feature_extractor_ptr_->getKnownLabel2IndexMap());
    transition_system_ptr_->setHeadWordRule(std::shared_ptr<HeadWordRule>(new HeadWordRule(feature_extractor_ptr_->getTagDict(), feature_extractor_ptr_->getLabelDict())));
    std::clog << "## End to init the transition system!" << std::endl;

    std::clog << "## Begin to generate the training examples..." << std::endl;
    GreedyChunker greedy_chunker;
    greedy_chunker.generateTrainingExamples(transition_system_ptr_.get(),
                                            feature_extractor_ptr_.get(),
                                            static_cast<SeqLabelerDataSet*>(training_set_ptr),
                                            greedy_example_ptrs_);
    std::clog << "### Greedy training examples' size: " << greedy_example_ptrs_.size() << std::endl;
    std::clog << "## End to generate the training examples!" << std::endl;
}
