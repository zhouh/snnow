/*************************************************************************
	> File Name: src/depparser/DepParseFeatureExtractor.cpp
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
 ************************************************************************/



#include "DepParseFeatureExtractor.h"

std::string DepParseFeatureExtractor::word_string = "word";
std::string DepParseFeatureExtractor::tag_string = "tag";
std::string DepParseFeatureExtractor::label_string = "label";
/**
 *   prepare the dictionary for feature extracting.
 *   The index of the dic here are independent
 */
void DepParseFeatureExtractor::getDictionaries(DataSet& d) {

	DepParseDataSet& data = static_cast<DepParseDataSet&>(d);

	StringSet labelSet;
	StringSet tagSet;
	StringSet wordSet;

	// insert to the set
    int data_size = data.getSize();
	for (unsigned i = 0; i < data_size; i++) {

        DepParseTree& gold_tree_i = static_cast<DepParseTree&>(*(data.outputs[i]));

		for (int j = 0; j < gold_tree_i.size; j++) { // the first node is -ROOT- node, skip
			auto tree_node = gold_tree_i.nodes[j];
			labelSet.insert(tree_node.label);
			tagSet.insert(tree_node.tag);
			wordSet.insert(tree_node.word);
		}
	}

    /*
     * initialize the dictionary table
     */
    dictionary_ptrs_table.resize(3, nullptr);
    dictionary_ptrs_table[c_word_dict_index].reset(new Dictionary(wordSet, word_string));
    dictionary_ptrs_table[c_tag_dict_index].reset(new Dictionary(tagSet, tag_string));
    dictionary_ptrs_table[c_dep_label_dict_index].reset(new Dictionary(labelSet, label_string));

}

FeatureVector DepParseFeatureExtractor::getFeatureVectors(
        State& base_state,
        Input& base_input ) {

	FeatureVector features;
	features.resize(3, feature_nums);

    DepParseState& state = static_cast<DepParseState&>(base_state);
    DepParseInput& input = static_cast<DepParseInput&>(base_input);
    
    input.word_cache;
    input.tag_cache;
    
	// positions 0-17 hold fWord, 18-35 hold fPos, 36-47 hold fLabel
//	int POS_OFFSET = 18;
//	int DEP_OFFSET = 36;
//	int STACK_OFFSET = 6;
//	int STACK_NUMBER = 6;

//	int index = 0;
	int word_index = 0;
	int tag_index = 0;
	int label_index = 0;

	int s0 = state.stacktop();
	int s1 = state.stack2top();
	int s2 = state.stack3top();
	int q0 = state.m_nNextWord >= state.len_ ? -1 : state.m_nNextWord;
	int q1 =
			(state.m_nNextWord + 1) >= state.len_ ?
					-1 : (state.m_nNextWord + 1);
	int q2 =
			(state.m_nNextWord + 2) >= state.len_ ?
					-1 : (state.m_nNextWord + 2);

	// words 
    // 0 - 12
	features[c_word_dict_index][tag_index++] = getWordIndex(s0, input.word_cache); // 0
	features[c_word_dict_index][word_index++] = getWordIndex(s1, input.word_cache); // 1
	features[c_word_dict_index][word_index++] = getWordIndex(s2, input.word_cache); // 2
	features[c_word_dict_index][word_index++] = getWordIndex(q0, input.word_cache); // 3
	features[c_word_dict_index][word_index++] = getWordIndex(q1, input.word_cache); // 4
	features[c_word_dict_index][word_index++] = getWordIndex(q2, input.word_cache); // 5

	features[c_tag_dict_index][tag_index++] = getTagIndex(s0, input.tag_cache);   // 6
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1, input.tag_cache);   // 7
	features[c_tag_dict_index][tag_index++] = getTagIndex(s2, input.tag_cache);   // 8
	features[c_tag_dict_index][tag_index++] = getTagIndex(q0, input.tag_cache);   // 9
	features[c_tag_dict_index][tag_index++] = getTagIndex(q1, input.tag_cache);   // 10
	features[c_tag_dict_index][tag_index++] = getTagIndex(q2, input.tag_cache);   // 11

	// s0l s0r s0l2 s0r2 s0ll s0rr
	int s0l, s0r, s0l2, s0r2, s0ll, s0rr;

#ifdef DEBUG
    std::cout<<"s0:"<<s0<<std::endl;
#endif
	s0l = s0 == -1 ? -1 : state.leftdep(s0);
	s0r = s0 == -1 ? -1 : state.rightdep(s0);
	s0l2 = s0 == -1 ? -1 : state.left2dep(s0);
	s0r2 = s0 == -1 ? -1 : state.left2dep(s0);
	s0ll = s0l == -1 ? -1 : state.leftdep(s0l);
	s0rr = s0r == -1 ? -1 : state.rightdep(s0r);
    
    //12 - 29
	features[c_word_dict_index][word_index++] = getWordIndex(s0l, input.word_cache);   // 12
	features[c_word_dict_index][word_index++] = getWordIndex(s0r, input.word_cache);   // 13
	features[c_word_dict_index][word_index++] = getWordIndex(s0l2, input.word_cache);  // 14
	features[c_word_dict_index][word_index++] = getWordIndex(s0r2, input.word_cache);  // 15
	features[c_word_dict_index][word_index++] = getWordIndex(s0ll, input.word_cache);  // 16
	features[c_word_dict_index][word_index++] = getWordIndex(s0rr, input.word_cache);  // 17

	features[c_tag_dict_index][tag_index++] = getTagIndex(s0l, input.tag_cache);     // 18
	features[c_tag_dict_index][tag_index++] = getTagIndex(s0r, input.tag_cache);     // 19
	features[c_tag_dict_index][tag_index++] = getTagIndex(s0l2, input.tag_cache);    // 20
	features[c_tag_dict_index][tag_index++] = getTagIndex(s0r2, input.tag_cache);    // 21
	features[c_tag_dict_index][tag_index++] = getTagIndex(s0ll, input.tag_cache);    // 22
	features[c_tag_dict_index][tag_index++] = getTagIndex(s0rr, input.tag_cache);    // 23
    //std::cout<<tag_index<<" "<<s0rr<<" "<<getTagIndex(s0rr, input.tag_cache)<<std::endl;

	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0l, &state);  // 24
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0r, &state);  // 25
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0l2, &state); // 26
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0r2, &state); // 27
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0ll, &state); // 28
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s0rr, &state); // 29

	// s1l s1r s1l2 s1r2 s1ll s1rr
	int s1l, s1r, s1l2, s1r2, s1ll, s1rr;
	s1l = s1 == -1 ? -1 : state.leftdep(s1);
	s1r = s1 == -1 ? -1 : state.rightdep(s1);
	s1l2 = s1 == -1 ? -1 : state.left2dep(s1);
	s1r2 = s1 == -1 ? -1 : state.left2dep(s1);
	s1ll = s1l == -1 ? -1 : state.leftdep(s1l);
	s1rr = s1r == -1 ? -1 : state.rightdep(s1r);

    // 30 - 47
	features[c_word_dict_index][word_index++] = getWordIndex(s1l, input.word_cache);
	features[c_word_dict_index][word_index++] = getWordIndex(s1r, input.word_cache);
	features[c_word_dict_index][word_index++] = getWordIndex(s1l2, input.word_cache);
	features[c_word_dict_index][word_index++] = getWordIndex(s1r2, input.word_cache);
	features[c_word_dict_index][word_index++] = getWordIndex(s1ll, input.word_cache);
	features[c_word_dict_index][word_index++] = getWordIndex(s1rr, input.word_cache);

	features[c_tag_dict_index][tag_index++] = getTagIndex(s1l, input.tag_cache);
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1r, input.tag_cache);
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1l2, input.tag_cache);
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1r2, input.tag_cache);
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1ll, input.tag_cache);
	features[c_tag_dict_index][tag_index++] = getTagIndex(s1rr, input.tag_cache);

	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1l, &state);
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1r, &state);
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1l2, &state);
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1r2, &state);
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1ll, &state);
	features[c_dep_label_dict_index][label_index++] = getLabelIndex(s1rr, &state);

    return features;

}

/**
 * generate the training examples for the greedy
 */
void DepParseFeatureExtractor::generateGreedyTrainingExamples(
        DepArcStandardSystem* transit_system_ptr,
        DepParseDataSet& training_data,
        std::vector<std::shared_ptr<Example>>& examples){

	auto & trees = training_data.outputs;
	auto & inputs = training_data.inputs;

	examples.clear();

	//for every sentence, cache the word and tag hash index
	for (unsigned i = 0; i < training_data.getSize(); i++) {


        auto & input_i = static_cast<DepParseInput&>(*(inputs[i]));
        auto & tree_i = static_cast<DepParseTree&>(*(trees[i]));
        // n shift and n reduce, one more reduce action for root
        int total_act_num_one_sentence = ( input_i.size() - 1 ) * 2;

        // in our current code, we do not cache the label set
//        /*
//         * cache the dependency label in the training set
//         */
//        std::vector<int> labelIndexCache(tree_i.size);
//        int index = 0;
//        for (auto iter = tree_i.nodes.begin(); iter != tree_i.nodes.end();
//             iter++) {
//            int labelIndex = getLabelIndex(iter->label);
//
//            if (labelIndex == -1) {
//                std::cerr << "Dep label " << iter->label
//                << " is not in labelMap!" << std::endl;
//                exit(1);
//            }
//
//            labelIndexCache[index] = labelIndex;
//            index++;
//        }

        //get state features
        std::vector<int> acts(total_act_num_one_sentence); //gold acts sequence for global

        std::shared_ptr<DepParseState> state_ptr;
        state_ptr.reset(new DepParseState());

        state_ptr->len_ = input_i.size();
        state_ptr->initCache();
        getCache(input_i);

        //for every state of a sentence
        for (int j = 0; !state_ptr->complete(); j++) {

            std::vector<int> labels;  // labels will be resized in function getValidActs()

            //get current state features
            FeatureVector fv = getFeatureVectors(*state_ptr, input_i);

            //get current state valid actions
            transit_system_ptr->getValidActs(*state_ptr, labels);

            //find gold action and move to next
            auto gold_act = transit_system_ptr->StandardMove(*state_ptr, tree_i);
            int gold_act_id = gold_act->getActionCode();

            transit_system_ptr->Move(state_ptr.operator*(), *gold_act);

            labels[ gold_act_id ] = 1;

            std::shared_ptr<Example> example_ptr(new Example( fv ,labels ));

            examples.push_back(example_ptr);
        }

    }

}

//
///**
// *   generate training examples for global learning
// *   assign the global example to gExamples, which is a member of class Depparser
// */
//void DepParseFeatureExtractor::generateTrainingExamples( ArcStandardSystem * tranSystem, std::vector<Instance>& instances,
//		std::vector<DepTree>& goldTrees, std::vector<GlobalExample>& gExamples ) {
//
//    gExamples.clear();
//
//	//for every sentence, cache the word and tag hash index
//	for (unsigned i = 0; i < instances.size(); i++) {
//
//        Instance & inst = instances[i];
//		auto & input = inst.input;
//		auto & gTree = goldTrees[i];
//		int actNum = ( input.size() - 1 ) * 2; // n shift and n reduce
//									   // one more reduce action for root
//
//        /*
//         * cache the dependency label in the training set
//         */
//		std::vector<int> labelIndexCache(gTree.size);
//		int index = 0;
//		for (auto iter = gTree.nodes.begin(); iter != gTree.nodes.end();
//				iter++) {
//            int labelIndex = getLabel(iter->label);
//
//			if (labelIndex == -1) {
//				std::cerr << "Dep label " << iter->label
//						<< " is not in labelMap!" << std::endl;
//				exit(1);
//			}
//
//			labelIndexCache[index] = labelIndex;
//			index++;
//		}
//
//		//get state features
//		std::vector<int> acts(actNum); //gold acts
//		std::vector<Example> examples; //features and labels
//
//		State* state = new State();
//        state.len_ = input.size();
//        state.initCache();
//        getCache(inst);
//
//		//for every state of a sentence
//		for (int j = 0; !state.complete(); j++) {
//			std::vector<int> features(featureNum);
//			std::vector<int> labels(tranSystem->nActNum, 0);
//
//			//get current state features
//			featureExtract(state, inst.wordCache, inst.tagCache, features);
//
//			//get current state valid actions
//			tranSystem->getValidActs(*state, labels);
//
//            /*
//             * display valid actions
//             */
//            /*for(int va = 0; va < labels.size(); va++){*/
//                //if( labels[va] < 0 )
//                    //continue;
//                //if( va == tranSystem->nLeftFirst || va == tranSystem->nRightFirst )
//                    //std::cout<<std::endl;
//
//                //std::cout<<tranSystem->DecodeUnlabeledAction(va)<<"+"<<tranSystem->DecodeLabel(va)<<" ";
//
//            //}
//
//			//find gold action and move to next
//			int goldAct = tranSystem->StandardMove(*state, gTree, labelIndexCache);
//			acts[j] = goldAct;
//
//            //std::cout << "move action:  "<< goldAct<<":"<<tranSystem->DecodeUnlabeledAction(goldAct)<<"+"<<tranSystem->DecodeLabel(goldAct) << std::endl;
//			tranSystem->Move(*state, goldAct);
///*            state.display();*/
//            //state.dispalyCache();
//
//			labels[goldAct] = 1;
//			Example example(features, labels);
//			examples.push_back(example);
//		}
//
//        GlobalExample ge_i( examples, acts, inst );
//
//		gExamples.push_back(ge_i);
//
//		delete state;
//
//#ifdef DEBUG
//        std::cout << "========================================"<< std::endl;
//#endif
//
//	}
//}


