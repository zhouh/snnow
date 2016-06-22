/*************************************************************************
	> File Name: ChunkerEvalb.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 21 Jun 2016 03:25:42 PM CST
 ************************************************************************/
#ifndef SNNOW_CHUNKEREVALB_H
#define SNNOW_CHUNKEREVALB_H

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <assert.h>

#include "SeqLabelerDataSet.h"

class Evaluation {
public:
    struct Metric {
        int correct_num;
        int gold_num;
        int guessed_num;

        Metric() {
            correct_num = 0;
            gold_num = 0;
            guessed_num = 0;
        }

        Metric(const Metric &m) {
            this->correct_num = m.correct_num;
            this->gold_num = m.gold_num;
            this->guessed_num = m.guessed_num;
        }

        Metric& operator= (const Metric &m) {
            if (this == &m) {
                return *this;
            }

            this->correct_num = m.correct_num;
            this->gold_num = m.gold_num;
            this->guessed_num = m.guessed_num;

            return *this;
        }
    };
public:
    std::vector<std::string> types;

    std::vector<Metric> metricOfTypes;

    Evaluation() {
        types = {"ADJP", "ADVP", "CONJP", "INTJ", "LST", "NP", "PP", "PRT", "SBAR", "UCP", "VP"};

        metricOfTypes.resize(types.size());
    }

    ~Evaluation() {}

    Evaluation(const Evaluation &e) {
        this->types = e.types;

        metricOfTypes.resize(this->types.size());
        for (int i = 0; i < this->types.size(); i++) {
            this->metricOfTypes[i] = e.metricOfTypes[i];
        }
    }

    Evaluation& operator= (const Evaluation &e) {
        if (this == &e) {
            return *this;
        }

        this->types = e.types;

        metricOfTypes.resize(this->types.size());
        for (int i = 0; i < this->types.size(); i++) {
            this->metricOfTypes[i] = e.metricOfTypes[i];
        }

        return *this;
    }

    void mergeEvaluation(Evaluation &eval) {
        assert (types.size() == eval.types.size());

        for (int i = 0; i < types.size(); i++) {
            this->metricOfTypes[i].correct_num += eval.metricOfTypes[i].correct_num;
            this->metricOfTypes[i].guessed_num += eval.metricOfTypes[i].guessed_num;
            this->metricOfTypes[i].gold_num    += eval.metricOfTypes[i].gold_num;
        }
    }

    void addGuessedCount(int idx, int count) {
        metricOfTypes[idx].guessed_num += count;
    }

    void addGuessedCount(const std::string &type, int count) {
        int idx = type2Idx(type);

        metricOfTypes[idx].guessed_num += count;
    }

    void addGoldCount(int idx, int count) {
        metricOfTypes[idx].gold_num += count;
    }

    void addGoldCount(const std::string &type, int count) {
        int idx = type2Idx(type);

        metricOfTypes[idx].gold_num += count;
    }

    void addCorrectCount(int idx, int count) {
        metricOfTypes[idx].correct_num += count;
    }

    void addCorrectCount(const std::string &type, int count) {
        int idx = type2Idx(type);

        metricOfTypes[idx].correct_num += count;
    }

    std::tuple<double, double, double> getMetricOf(int idx) {
        double precision, recall, f_score;
        if (metricOfTypes[idx].guessed_num != 0){
            precision = (double)metricOfTypes[idx].correct_num / metricOfTypes[idx].guessed_num;
        } else {
            precision = 0.0;
        }

        if (metricOfTypes[idx].gold_num != 0) {
            recall = (double)metricOfTypes[idx].correct_num / metricOfTypes[idx].gold_num;
        } else {
            recall = 0.0;
        }

        if (precision + recall == 0.0) {
            f_score = 0.0;
        } else {
            f_score = 2 * precision * recall / (precision + recall);
        }

        auto res = std::make_tuple(100.0 * precision, 100.0 * recall, 100.0 * f_score);

        return res;
    }

    std::tuple<double, double, double> getMetricOf(const std::string &type) {
        int idx = type2Idx(type);

        return getMetricOf(idx);
    }

    int type2Idx(const std::string &type) {
        for (int i = 0; i < types.size(); i++) {
            if (types[i] == type) {
                return i;
            }
        }

        std::cerr << "In evaluation type2Idx function, type " << type << " can not be found!" << std::endl;
        exit(0);
    }

    std::tuple<double, double, double> getTotalMetric() {
        int correct_num = 0;
        int gold_num = 0 ;
        int guessed_num = 0;

        for (int i = 0; i < types.size(); i++) {
            Metric &m = metricOfTypes[i];

            correct_num += m.correct_num;
            guessed_num += m.guessed_num;
            gold_num += m.gold_num;
        }

        double precision, recall, f_score;
        if (guessed_num != 0){
            precision = (double)correct_num / guessed_num;
        } else {
            precision = 0.0;
        }

        if (gold_num != 0) {
            recall = (double)correct_num / gold_num;
        } else {
            recall = 0.0;
        }

        if (precision + recall == 0.0) {
            f_score = 0.0;
        } else {
            f_score = 2 * precision * recall / (precision + recall);
        }

        return std::make_tuple(100.0 * precision, 100.0 * recall, 100.0 * f_score);
    }

    void print() {
        auto sf = std::cerr.flags();
        auto sp = std::cerr.precision();
        std::cerr.flags(std::ios::fixed);
        std::cerr.precision(4);

        int total_guessed = 0, total_gold = 0, total_correct = 0;
        std::cerr << std::left << std::setw(8) << "type" << "\t";
        std::cerr << std::left << std::setw(8) << "corrNum" << "\t";
        std::cerr << std::left << std::setw(8) << "predNum" << "\t";
        std::cerr << std::left << std::setw(8) << "gold_num" << "\t";
        std::cerr << std::left << std::setw(8) << "prec" << "\t";
        std::cerr << std::left << std::setw(8) << "recall" << "\t";
        std::cerr << std::left << std::setw(8) << "F-score" << std::endl;
        for (int i = 0; i < types.size(); i++) {
            total_guessed += metricOfTypes[i].guessed_num;
            total_gold += metricOfTypes[i].gold_num;
            total_correct += metricOfTypes[i].correct_num;

            double precision, recall, f_score;
            if (metricOfTypes[i].guessed_num != 0){
                precision = (double)metricOfTypes[i].correct_num / metricOfTypes[i].guessed_num;
            } else {
                precision = 0.0;
            }

            if (metricOfTypes[i].gold_num != 0) {
                recall = (double)metricOfTypes[i].correct_num / metricOfTypes[i].gold_num;
            } else {
                recall = 0.0;
            }

            if (precision + recall == 0.0) {
                f_score = 0.0;
            } else {
                f_score = 2 * precision * recall / (precision + recall);
            }

            std::cerr << std::left << std::setw(8) << types[i] << "\t";
            std::cerr << std::left << std::setw(8) << metricOfTypes[i].correct_num << "\t";
            std::cerr << std::left << std::setw(8) << metricOfTypes[i].guessed_num << "\t";
            std::cerr << std::left << std::setw(8) << metricOfTypes[i].gold_num << "\t";
            std::cerr << std::left << std::setw(8) << precision << "\t";
            std::cerr << std::left << std::setw(8) << recall << "\t";
            std::cerr << std::left << std::setw(8) << f_score << std::endl;
        }

        double precision, recall, f_score;
        if (total_guessed != 0){
            precision = (double)total_correct / total_guessed;
        } else {
            precision = 0.0;
        }

        if (total_gold != 0) {
            recall = (double)total_correct / total_gold;
        } else {
            recall = 0.0;
        }

        if (precision + recall == 0.0) {
            f_score = 0.0;
        } else {
            f_score = 2 * precision * recall / (precision + recall);
        }

        std::cerr << std::left << std::setw(8) << "total" << "\t";
        std::cerr << std::left << std::setw(8) << total_correct << "\t";
        std::cerr << std::left << std::setw(8) << total_guessed << "\t";
        std::cerr << std::left << std::setw(8) << total_gold << "\t";
        std::cerr << std::left << std::setw(8) << precision << "\t";
        std::cerr << std::left << std::setw(8) << recall << "\t";
        std::cerr << std::left << std::setw(8) << f_score << std::endl;

        std::cerr.flags(sf);
        std::cerr.precision(sp);
    }
    // S-ADJP: 15
    // S-ADVP: 16
    // S-CLP: 17
    // S-DNP: 18
    // S-DP: 19
    // S-DVP: 20
    // S-LCP: 21
    // S-LST: 22
    // S-NP: 23
    // S-PP: 24
    // S-QP: 25
    // S-VP: 26
};

class ChunkerEvalb {
    // return (precision, recall, FB1)
    static Evaluation eval(SeqLabelerDataSet &predicts, SeqLabelerDataSet &golds) {
        assert(predicts.getSize() == golds.getSize());

        Evaluation evaluation;
        for (int i = 0; i < predicts.getSize(); i++){
            RawSequence pscs = transformEtypeFormat2BIOFormat(predicts.raw_sequences_[i]);
            RawSequence gscs = transformEtypeFormat2BIOFormat(golds.raw_sequences_[i]);

            auto res = eval(pscs, gscs);

            evaluation.mergeEvaluation(res);
        }
        return evaluation;
    }

    // return: (correct_count, gold_count, predict_count)
    static Evaluation eval(RawSequence &predict, RawSequence &gold) {
        assert (predict.size() == gold.size());

        Evaluation evaluation;
        using std::string;

        const std::vector<std::vector<std::string>> &correct_terms = gold;
        const std::vector<std::vector<std::string>> &guessed_terms = predict;

        std::vector<std::tr1::unordered_set<string>> guessed_sets(evaluation.types.size());
        std::vector<std::tr1::unordered_set<string>> gold_sets(evaluation.types.size());

        int length = predict.size();
        string correct_label, guessed_label;

        string lastLabel = "O";
        int lastStart = -1;
        int lastEnd = -1;
        for (int i = 0; i < length; i++) {
            guessed_label = guessed_terms[i][2];
            if (guessed_label[0] == 'B' || guessed_label[0] == 'O') {
                if (lastLabel != "O") {
                    std::string label_type = lastLabel.substr(2);
                    int idx = evaluation.type2Idx(label_type);

                    guessed_sets[idx].insert(label_type + "_" + std::to_string(lastStart) + "_" + std::to_string(lastEnd));
                }

                lastStart = i;
                lastEnd = i;
                lastLabel = guessed_label;
            }

            lastEnd = i;
        }
        if (lastLabel != "O") {
            if (lastLabel != "O") {
                std::string label_type = lastLabel.substr(2);
                int idx = evaluation.type2Idx(label_type);

                guessed_sets[idx].insert(label_type + "_" + std::to_string(lastStart) + "_" + std::to_string(lastEnd));
            }
        }

        lastLabel = "O";
        lastStart = -1;
        lastEnd = -1;
        for (int i = 0; i < length; i++) {
            correct_label = correct_terms[i][2];

            if (correct_label[0] == 'B' || correct_label[0] == 'O') {
                if (lastLabel != "O") {
                    std::string label_type = lastLabel.substr(2);
                    int idx = evaluation.type2Idx(label_type);

                    gold_sets[idx].insert(label_type + "_" + std::to_string(lastStart) + "_" + std::to_string(lastEnd));
                }

                lastStart = i;
                lastEnd = i;
                lastLabel = correct_label;
            }
            lastEnd = i;
        }
        if (lastLabel != "O") {
            if (lastLabel != "O") {
                std::string label_type = lastLabel.substr(2);
                int idx = evaluation.type2Idx(label_type);

                gold_sets[idx].insert(label_type + "_" + std::to_string(lastStart) + "_" + std::to_string(lastEnd));
            }
        }

        for (int i = 0; i < evaluation.types.size(); i++) {
            std::tr1::unordered_set<std::string> &guessed_set = guessed_sets[i];
            std::tr1::unordered_set<std::string> &gold_set = gold_sets[i];
            int found_gold_num = gold_set.size();
            int found_guessed_num = guessed_set.size();
            evaluation.addGoldCount(i, found_gold_num);
            evaluation.addGuessedCount(i, found_guessed_num);
            for (const string &s : guessed_set) {
                if (gold_set.find(s) != gold_set.end()) {
                    evaluation.addCorrectCount(i, 1);
                }
            }
        }

        return evaluation;
    }

private:
    // startOfChunk: checks if a chunk started between the previous and current word
    // arguments:    previous and current chunk tags, previous and current types
    // note:         this code is not capable of handling other chunk representations
    //               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    //               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    static bool startOfChunk(std::string &prev_tag, std::string &tag, std::string &prev_type, std::string &type) {
        bool chunk_start = false;

        if (prev_tag == "B") {
            if (tag == "B") {
                chunk_start = true;
            }
        } else if (prev_tag == "I") {
            if (tag == "B") {
                chunk_start = true;
            }
        } else {
            if (tag == "B" || tag == "I") {
                chunk_start = true;
            }
        }

        if (tag != "O" && tag != "." && prev_type != type) {
            chunk_start = true;
        }

        return chunk_start;
    }

    // endOfChunk: checks if a chunk ended between the preivous and current word
    // arguments:  previous and current chunk tags, previous and current types
    // note:       this code is not capable of handling other chunk representations
    //             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    //             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    static bool endOfChunk(std::string &prev_tag, std::string &tag, std::string prev_type, std::string &type) {
        bool chunk_end = false;

        if (prev_tag == "B") {
            if (tag == "B" || tag == "O") {
                chunk_end = true;
            }
        } else if (prev_tag == "I") {
            if (tag == "B" || tag == "O") {
                chunk_end = true;
            }
        }

        if (prev_tag != "O" && prev_tag != "." && prev_type != type) {
            chunk_end = true;
        }

        return chunk_end;
    }
};

#endif // SNNOW_CHUNKEREVALB_H
