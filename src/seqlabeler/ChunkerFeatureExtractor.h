/*************************************************************************
	> File Name: ChunkerFeatureExtractor.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 15 Jun 2016 10:59:27 PM CST
 ************************************************************************/
#ifndef SNNOW_CHUNKERFEATUREEXTRACTOR_H
#define SNNOW_CHUNKERFEATUREEXTRACTOR_H

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream>
#include <gflags/gflags.h>

#include "base/FeatureExtractor.h"
#include "Dict.h"
#include "ChunkerMacro.h"
#include "SeqLabelerDataSet.h"
#include "FeatureType.h"
#include "ChunkerMacro.h"
#include "ChunkerState.h"
#include "SeqLabelerInput.h"
#include "SeqLabelerOutput.h"

DECLARE_string(dict_file);
DECLARE_string(test_dict_file);

class ChunkerFeatureExtractor : public FeatureExtractor {
public:
    const int c_word_dict_index_ = 0;
    const int c_tag_dict_index_ = 1;
    const int c_label_dict_index_ = 2;
    const int c_capital_dict_index_ = 3;
    const int c_affix_dict_index_ = 4;

    std::string word_string_;
    std::string tag_string_;
    std::string label_string_;
    std::string capital_string_;
    std::string affix_string_;

public:
    std::vector<int> feature_nums_;
    DictionaryVectorPtrs dictionary_ptrs_table_;
    FeatureTypes feature_types_;

public:
    ChunkerFeatureExtractor() {
        word_string_ = "word";
        tag_string_ = "tag";
        label_string_ = "label";
        capital_string_ = "capital";
        affix_string_ = "affix";

        feature_nums_ = {5, 16, 5, 5, 6};
    }

    ~ChunkerFeatureExtractor() = default;

    void getDictionaries(DataSet *data_ptr) {
        using namespace std;

        SeqLabelerDataSet &dataset = *(static_cast<SeqLabelerDataSet *>(data_ptr));

        StringSet word_table;
        StringSet affix_table;
        string line;
        ifstream is(FLAGS_dict_file);
        while (!is.eof()) {
            getline(is, line);

            string processed_word = processWord(line);

            if (processed_word == "") {
                continue;
            }

            word_table.insert(processed_word);
            for (const std::string &affix : getAffixes(processed_word)) {
                affix_table.insert(affix);
            }
        }

        StringSet wordset;
        StringSet tagset;
        StringSet labelset;
        StringSet affixset;

        StringSet uni_tagset;
        for (vector<vector<string>> &seq : dataset.raw_sequences_) {
            for (vector<string> &term : seq) {
                string &word = term[0];
                string &tag = term[1];
                string &label = term[2];

                std::string processed_word = processWord(word);
                if (word_table.find(processed_word) != word_table.end()) {
                    wordset.insert(processed_word);
                }
                uni_tagset.insert(tag);
                labelset.insert(label);
                for (const std::string &affix : getAffixes(processed_word)) {
                    if (affix_table.find(affix) != affix_table.end()) {
                        affixset.insert(affix);
                    }
                }
            }
        }

        ifstream testis(FLAGS_test_dict_file);
        while (!testis.eof()) {
            getline(testis, line);

            string processed_word = processWord(line);

            if ((word_table.find(processed_word) != word_table.end()) && processed_word != "") {
                wordset.insert(processed_word);
            }
        }

        // construct POS-tag feature(unigram and bigram)
        for (const string &uni_tag : uni_tagset) {
            tagset.insert(uni_tag);

            tagset.insert("NULLPOS-" + uni_tag);
            tagset.insert(uni_tag + "-NULLPOS");
        }
        tagset.insert("NULLPOS-NULLPOS");
        for (const string &uni1 : uni_tagset) {
            for (const string &uni2 : uni_tagset) {
                tagset.insert(uni1 + "-" + uni2);
            }
        }

        dictionary_ptrs_table_.resize(feature_nums_.size(), nullptr);

        dictionary_ptrs_table_[c_word_dict_index_].reset(new Dictionary(wordset, word_string_));
        dictionary_ptrs_table_[c_tag_dict_index_].reset(new Dictionary(tagset, tag_string_));
        dictionary_ptrs_table_[c_label_dict_index_].reset(new Dictionary(labelset, label_string_));

        StringSet capitalset;
        capitalset.insert(noncapitalstr);
        capitalset.insert(allcapitalstr);
        capitalset.insert(firstlettercapstr);
        capitalset.insert(hadonecapstr);
        dictionary_ptrs_table_[c_capital_dict_index_].reset(new Dictionary(capitalset, capital_string_));

        dictionary_ptrs_table_[c_affix_dict_index_].reset(new Dictionary(affixset, affix_string_));
    }

    void setFeatureTypes() {
        FeatureType word_feat_type(word_string_,
                                   feature_nums_[c_word_dict_index_],
                                   dictionary_ptrs_table_[c_word_dict_index_]->size(),
                                   c_word_feature_dim);

        FeatureType tag_feat_type(tag_string_,
                                  feature_nums_[c_tag_dict_index_],
                                  dictionary_ptrs_table_[c_tag_dict_index_]->size(),
                                  c_tag_feature_dim);

        FeatureType label_feat_type(label_string_,
                                    feature_nums_[c_label_dict_index_],
                                    dictionary_ptrs_table_[c_label_dict_index_]->size(),
                                    c_label_feature_dim);

        FeatureType capital_feat_type(capital_string_,
                                      feature_nums_[c_capital_dict_index_],
                                      dictionary_ptrs_table_[c_capital_dict_index_]->size(),
                                      c_capital_feature_dim );

        FeatureType affix_feat_type(affix_string_,
                                    feature_nums_[c_affix_dict_index_],
                                    dictionary_ptrs_table_[c_affix_dict_index_]->size(),
                                    c_affix_feature_dim);

        feature_types_.push_back(word_feat_type);
        feature_types_.push_back(tag_feat_type);
        feature_types_.push_back(label_feat_type);
        feature_types_.push_back(capital_feat_type);
        feature_types_.push_back(affix_feat_type);
    }

    void displayDict() {
        std::clog << "### knownWords   size: " << dictionary_ptrs_table_[c_word_dict_index_]->size() << std::endl;
        std::clog << "### knownTags    size: " << dictionary_ptrs_table_[c_tag_dict_index_]->size() << std::endl;
        std::clog << "### knownLabels  size: " << dictionary_ptrs_table_[c_label_dict_index_]->size() << std::endl;
        std::clog << "### knownLabel vector: " << std::endl;
        const StringArray &knownLabels = dictionary_ptrs_table_[c_label_dict_index_]->getKnownStringVector();
        for (int i = 0; i < knownLabels.size(); i++) {
            std::clog << "    " << knownLabels[i] << ": " << i << std::endl;
        }

        std::clog << "### knownCapitals size: " << dictionary_ptrs_table_[c_capital_dict_index_]->size() << std::endl;
        std::clog << "### knownAffixes  size: " << dictionary_ptrs_table_[c_affix_dict_index_]->size() << std::endl;
        // std::clog << "### knownlabels map: " << std::endl;
        // const String2IndexMap &labelMap = dictionary_ptrs_table_[c_label_dict_index_]->getMap();
        // for (auto &e : labelMap) {
        //     std::clog << "    " << e.first << ": " << e.second << std::endl;
        // }
    }

    void displayFeatureTypes() {
        std::clog << "### featuretype vector: " << std::endl;
        for (const FeatureType &ft : feature_types_) {
            std::clog << "###   typename:  " << ft.type_name << std::endl;
            std::clog << "###     featsize:  " << ft.feature_size << std::endl;
            std::clog << "###     dictsize:  " << ft.dictionary_size << std::endl;
            std::clog << "###     embedsize: " << ft.feature_embedding_size << std::endl;
        }
    }

    void generateCache(SeqLabelerInput *input, SeqLabelerOutput *output, RawSequence &raw_seq) {
        const int seq_len = raw_seq.size();

        input->clear();
        output->clear();

        input->word_cache_.resize(seq_len);
        input->tag_cache_.resize(seq_len);
        input->bi_tag_cache_.resize(seq_len);
        input->capital_cache_.resize(seq_len);
        input->affix_cache_.resize(seq_len);
        output->output_label_.resize(seq_len);

        for (int index = 0; index < seq_len; index++) {
            std::vector<std::string> &term = raw_seq[index];

            const int word_idx = getWordIndex(term[0]);
            input->word_cache_[index] = word_idx;

            const int uni_tag_idx = getUniTagIndex(term[1]);
            input->tag_cache_[index] = uni_tag_idx;

            using std::string;

            auto getPOS = [&raw_seq](int index, int length) -> std::string {
                if (index < 0 || index >= length) {
                    return std::string("NULLPOS");
                }

                return raw_seq[index][1];
            };

            string neg1POS, neg2POS, pos1POS, pos2POS;
            neg1POS = getPOS(index - 1, seq_len);
            neg2POS = getPOS(index - 2, seq_len);
            pos1POS = getPOS(index + 1, seq_len);
            pos2POS = getPOS(index + 2, seq_len);
            input->bi_tag_cache_[index].push_back(getBiTagIndex(neg2POS, neg1POS));
            input->bi_tag_cache_[index].push_back(getBiTagIndex(neg1POS, term[1]));
            input->bi_tag_cache_[index].push_back(getBiTagIndex(term[1], pos1POS));
            input->bi_tag_cache_[index].push_back(getBiTagIndex(pos1POS, pos2POS));

            input->affix_cache_[index] = getAffixIndexes(term[0]);
            input->capital_cache_[index] = getCapitalIndex(term[0]);

            output->output_label_[index] = getLabelIndex(term[2]);
        }
    }

    FeatureVector getFeatureVectors(State *state_ptr, Input *input_ptr) {

        ChunkerState &chunk_state = *(static_cast<ChunkerState *>(state_ptr));
        SeqLabelerInput &chunk_input = *(static_cast<SeqLabelerInput *>(input_ptr));

        auto extractWordFeature = [&chunk_state, &chunk_input, this]() -> std::vector<int> {
            auto getWordIndex = [&chunk_state, &chunk_input, this](int index) -> int {
                if (index < 0 || index >= chunk_state.sequenceLength()) {
                    return this->dictionary_ptrs_table_[c_word_dict_index_]->getNullIndex();
                }

                return chunk_input.word_cache_[index];
            };

            std::vector<int> features;

            int current_index = chunk_state.index_ + 1;

            int neg2UniWord   = getWordIndex(current_index - 2);
            int neg1UniWord   = getWordIndex(current_index - 1);
            int pos0UniWord   = getWordIndex(current_index);
            int pos1UniWord   = getWordIndex(current_index + 1);
            int pos2UniWord   = getWordIndex(current_index + 2);
            features.push_back(neg2UniWord);
            features.push_back(neg1UniWord);
            features.push_back(pos0UniWord);
            features.push_back(pos1UniWord);
            features.push_back(pos2UniWord);

            return features;
        };
        auto extractTagFeature = [&chunk_state, &chunk_input, this]() -> std::vector<int> {
            auto getTagIndex = [&chunk_state, &chunk_input, this](int index) -> int {
                if (index < 0 || index >= chunk_state.sequenceLength()) {
                    return this->dictionary_ptrs_table_[c_tag_dict_index_]->getNullIndex();
                }

                return chunk_input.tag_cache_[index];
            };

            std::vector<int> features;

            int current_index = chunk_state.index_ + 1;

            int neg2UniTag  = getTagIndex(current_index - 2);
            int neg1UniTag  = getTagIndex(current_index - 1);
            int pos0UniTag  = getTagIndex(current_index);
            int pos1UniTag  = getTagIndex(current_index + 1);
            int pos2UniTag  = getTagIndex(current_index + 2);
            features.push_back(neg2UniTag);
            features.push_back(neg1UniTag);
            features.push_back(pos0UniTag);
            features.push_back(pos1UniTag);
            features.push_back(pos2UniTag);

            int neg1StartTag  = getTagIndex(chunk_state.prev_chunk_index_);
            int neg1EndTag    = getTagIndex(chunk_state.curr_chunk_index_ - 1);
            int pos0StartTag  = getTagIndex(chunk_state.curr_chunk_index_);
            int pos0EndTag    = getTagIndex(chunk_state.ongo_chunk_index_ - 1);
            int onGoStartTag  = getTagIndex(chunk_state.ongo_chunk_index_);
            features.push_back(neg1StartTag);
            features.push_back(neg1EndTag);
            features.push_back(pos0StartTag);
            features.push_back(pos0EndTag);
            features.push_back(onGoStartTag);

            int prevHeadIndex = chunk_state.prev_head_index_;
            int currHeadIndex = chunk_state.curr_head_index_;
            features.push_back(getTagIndex(prevHeadIndex));
            features.push_back(getTagIndex(currHeadIndex));

            for (int bi_tag : chunk_input.bi_tag_cache_[current_index]) {
                features.push_back(bi_tag);
            }

            return features;
        };
        auto extractLabelFeature = [&chunk_state, &chunk_input, this]() -> std::vector<int> {
            auto getLabelIndex = [&chunk_state, &chunk_input, this](int index) -> int {
                if (index < 0 || index >= chunk_state.sequenceLength()) {
                    return this->dictionary_ptrs_table_[c_label_dict_index_]->getNullIndex();
                }

                return chunk_state.chunked_label_ids_[index];
            };

            std::vector<int> features;

            int current_index = chunk_state.index_ + 1;

            int neg2UniLabel  = getLabelIndex(current_index - 2);
            int neg1UniLabel  = getLabelIndex(current_index - 1);
            int prevEndLabel = getLabelIndex(chunk_state.curr_chunk_index_ - 1);
            int currEndLabel = getLabelIndex(chunk_state.ongo_chunk_index_ - 1);
            int ongoStartLabel = getLabelIndex(chunk_state.ongo_chunk_index_);

            features.push_back(neg2UniLabel);
            features.push_back(neg1UniLabel);
            features.push_back(prevEndLabel);
            features.push_back(currEndLabel);
            features.push_back(ongoStartLabel);

            return features;
        };
        auto extractCapitalFeature = [&chunk_state, &chunk_input, this]() -> std::vector<int> {
            auto getCapitalIndex = [&chunk_state, &chunk_input, this](int index) -> int {
                if (index < 0 || index >= chunk_state.sequenceLength()) {
                    return this->dictionary_ptrs_table_[c_capital_dict_index_]->getNullIndex();
                }

                return chunk_input.capital_cache_[index];
            };

            std::vector<int> features;

            int current_index = chunk_state.index_ + 1;

            int neg2UniCap = getCapitalIndex(current_index - 2);
            int neg1UniCap = getCapitalIndex(current_index - 1);
            int pos0UniCap = getCapitalIndex(current_index);
            int pos1UniCap = getCapitalIndex(current_index + 1);
            int pos2UniCap = getCapitalIndex(current_index + 2);

            features.push_back(neg2UniCap);
            features.push_back(neg1UniCap);
            features.push_back(pos0UniCap);
            features.push_back(pos1UniCap);
            features.push_back(pos2UniCap);

            return features;
        };
        auto extractAffixFeature = [&chunk_state, &chunk_input, this]() -> std::vector<int> {
            std::vector<int> features;

            int current_index = chunk_state.index_ + 1;

            for (int ai : chunk_input.affix_cache_[current_index]) {
                features.push_back(ai);
            }

            return features;
        };

        FeatureVector vec_of_features;

        vec_of_features.resize(feature_nums_);

        vec_of_features.setVector(c_word_dict_index_,  extractWordFeature());
        vec_of_features.setVector(c_tag_dict_index_,   extractTagFeature());
        vec_of_features.setVector(c_label_dict_index_, extractLabelFeature());
        vec_of_features.setVector(c_capital_dict_index_, extractCapitalFeature());
        vec_of_features.setVector(c_affix_dict_index_, extractAffixFeature());

        return vec_of_features;
    }

    int getTotalInputSize() {
        int retval = 0;
        for (int i = 0; i < feature_types_.size(); ++i) {
            retval += feature_types_[i].feature_embedding_size * feature_types_[i].feature_size;

        }
        return retval;
    }

    const std::shared_ptr<Dictionary>& getWordDict() {
        return dictionary_ptrs_table_[c_word_dict_index_];
    }

    const std::shared_ptr<Dictionary>& getTagDict() {
        return dictionary_ptrs_table_[c_tag_dict_index_];
    }

    const std::shared_ptr<Dictionary>& getLabelDict() {
        return dictionary_ptrs_table_[c_label_dict_index_];
    }

    int getWordIndex(const std::string &word) {
        std::string processed_word = processWord(word);

        return dictionary_ptrs_table_[c_word_dict_index_]->getStringIndex(processed_word);
    }

    int getUniTagIndex(const std::string &tag) {
        return dictionary_ptrs_table_[c_tag_dict_index_]->getStringIndex(tag);
    }

    int getBiTagIndex(const std::string &uni_tag1, const std::string &uni_tag2) {
        return dictionary_ptrs_table_[c_tag_dict_index_]->getStringIndex(uni_tag1 + "-" + uni_tag2);
    }

    int getLabelIndex(const std::string &label) {
        return  dictionary_ptrs_table_[c_label_dict_index_]->getStringIndex(label);
    }

    int getCapitalIndex(const std::string &word) {
        bool isNoncap = true;
        bool isAllcap = true;
        bool isFirstCap = false;
        bool isHadCap  = false;

        if (isupper(word[0])) {
            isFirstCap = true;
        }

        for (char ch : word) {
            if (isupper(ch)) {
                isHadCap = true;
                isNoncap = false;
            } else {
                isAllcap = false;
            }
        }

        if (isNoncap) {
            return dictionary_ptrs_table_[c_capital_dict_index_]->getStringIndex(noncapitalstr);
        }

        if (isAllcap) {
            return dictionary_ptrs_table_[c_capital_dict_index_]->getStringIndex(allcapitalstr);
        }

        if (isFirstCap) {
            return dictionary_ptrs_table_[c_capital_dict_index_]->getStringIndex(firstlettercapstr);
        }

        if (isHadCap) {
            return dictionary_ptrs_table_[c_capital_dict_index_]->getStringIndex(hadonecapstr);
        }

        std::cerr << "word2CapfeatIdx wrong: " << word << std::endl;
        exit(1);
    }

    std::string getLabelString(const int id) {
        return dictionary_ptrs_table_[c_label_dict_index_]->getString(id);
    }

    std::vector<int> getAffixIndexes(const std::string &word) {
        std::string processed_word = processWord(word);

        std::vector<std::string> affixes = getAffixes(processed_word);

        std::vector<int> ret;

        for (const std::string &affix : affixes) {
            ret.push_back(dictionary_ptrs_table_[c_affix_dict_index_]->getStringIndex(affix));
        }

        return ret;
    }

    const StringArray& getKnownLabelVector() {
        return dictionary_ptrs_table_[c_label_dict_index_]->getKnownStringVector();
    }

    const String2IndexMap& getKnownLabel2IndexMap() {
        return dictionary_ptrs_table_[c_label_dict_index_]->getMap();
    }

    void returnInput(std::vector<FeatureVector> &vec_of_featurevector,
                     std::vector<std::shared_ptr<FeatureEmbedding>> &vec_of_featemb_ptr,
                     TensorContainer<cpu, 2, real_t> &input){
        for(unsigned beam_index = 0; beam_index < static_cast<unsigned>(vec_of_featurevector.size()); beam_index++) { // for every beam item
            FeatureVector &featvec = vec_of_featurevector[beam_index];

            int inputIndex = 0;
            for (int feat_type_index = 0; feat_type_index < static_cast<int>(featvec.size()); feat_type_index++) {
                const std::vector<int> curr_featvec = featvec.getVector(feat_type_index);
                const int curr_featsize = feature_types_[feat_type_index].feature_size;
                const int curr_embsize  = feature_types_[feat_type_index].feature_embedding_size;
                std::shared_ptr<FeatureEmbedding> &curr_featemb = vec_of_featemb_ptr[feat_type_index];

                for (auto featId : curr_featvec) {
                    Copy(input[beam_index].Slice(inputIndex, inputIndex + curr_embsize),
                         curr_featemb->data[featId],
                         curr_featemb->data.stream_);
                    inputIndex += curr_embsize;
                }
            }
        }
    }
private:
    std::string processWord(const std::string &word) {
        // if (word == "-LRB-" || word == "-RRB-" || word == "-URL-") {
        //     return word;
        // }

        std::string low_word(word);

        std::transform(word.begin(), word.end(), low_word.begin(), ::tolower);

        low_word = replaceNumber(low_word);

        return low_word;
    }

    std::string replaceNumber(const std::string &word) {
        static const std::string numberstr = "NUMBER";

        std::string ret;

        bool isNumber = false;
        for (char ch : word) {
            if (isdigit(ch)) {
                if (isNumber){
                    continue;
                }

                isNumber = true;

                ret += numberstr;
            } else {
                if (isNumber && (ch == ',' || ch == '.' || ch == '%' || ch == '/')) {
                    continue;
                }

                isNumber = false;
                ret.push_back(ch);
            }
        }

        return ret;
    }

private:
    const std::string noncapitalstr = "-NONCAPITAL-";
    const std::string allcapitalstr = "-ALLCAPITAL-";
    const std::string firstlettercapstr = "-FIRSTLETTERCAPITAL-";
    const std::string hadonecapstr = "-HADONECAPITAL-";

private:
    std::vector<std::string> getAffixes(const std::string &word) {
        std::vector<std::string> rets;

        for (int len = 1; len <= 3; len++) {
            for (auto s : getAffixesByLen(word, len)) {
                rets.push_back(s);
            }
        }

        return rets;
    }

    std::vector<std::string> getAffixesByLen(const std::string &word, int len) {
        std::string w(word);

        std::transform(word.begin(), word.end(), w.begin(), ::tolower);

        std::vector<std::string> rets;

        std::string prefix = "", suffix = "";
        if (w.size() < len) {
            for (int ti = 0; ti < len - w.size(); ti++) {
                prefix.push_back('#');
                suffix.push_back('#');
            }
            prefix = prefix + w;
            suffix = w + suffix;
        } else {
            prefix = w.substr(0, len);
            suffix = w.substr(w.size() - len, len);
        }

        rets.push_back(prefix);
        rets.push_back(suffix);

        return rets;
    }
};

#endif // SNNOW_CHUNKERFEATUREEXTRACTOR_H
