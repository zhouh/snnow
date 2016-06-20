/*************************************************************************
	> File Name: SeqLabeler.cu
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 14 Jun 2016 03:33:34 PM CST
 ************************************************************************/
#include "SeqLabeler.h"

SeqLabeler::SeqLabeler(bool b_train) {
    b_train_ = b_train;

    feature_extractor_ptr_.reset(new ChunkerFeatureExtractor());
    transition_system_ptr_.reset(new ChunkerTransitionSystem());
}

void SeqLabeler::train(DataSet &training_set, DataSet &dev_set) {
    trainInit(training_set);
}

void SeqLabeler::greedyTrain(DataSet &training_set, DataSet &dev_set) {
}

double SeqLabeler::test(DataSet &test_data, Model <cpu> &model, FeedForwardNNet <gpu> &net) {

    return 0.0;
}

void SeqLabeler::trainInit(DataSet& training_set) {
    std::clog << "======================================"<<std::endl;
    std::clog << "# Training Init!" << std::endl;
    std::clog << "# Training Instance Num: " << training_set.getSize() << std::endl;
    std::clog << "======================================"<<std::endl;

    // prepare the handler for parsing
    std::clog << "## Begin to init the dictionaries!" << std::endl;
    feature_extractor_ptr_->getDictionaries(training_set);  // dictionary for feature index
    feature_extractor_ptr_->displayDict();

    std::clog << "## End to init the dictionaries!" << std::endl;

    std::clog << "## Begin to create feature types!" << std::endl;
    feature_extractor_ptr_->setFeatureTypes();
    feature_extractor_ptr_->displayFeatureTypes();
    std::clog << "## End to create feature types!" << std::endl;

    std::clog << "## Begin to init the transition system!" << std::endl;
    transition_system_ptr_->makeTransitions(feature_extractor_ptr_->getKnownLabelVector(), feature_extractor_ptr_->getKnownLabel2IndexMap());
    transition_system_ptr_->setHeadWordRule(std::shared_ptr<HeadWordRule>(new HeadWordRule(feature_extractor_ptr_->getTagDict(), feature_extractor_ptr_->getLabelDict())));
    std::clog << "## End to init the transition system!" << std::endl;
}
