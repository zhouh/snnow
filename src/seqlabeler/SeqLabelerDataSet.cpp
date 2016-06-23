/*************************************************************************
	> File Name: SeqLabelerDataSet.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Thu 16 Jun 2016 02:12:22 PM CST
 ************************************************************************/
#include "SeqLabelerDataSet.h"

std::ostream& operator<< (std::ostream &os, RawSequence &raw_sequence) {
    using namespace std;

    auto process_sequence = [&raw_sequence] {
        auto is_chunk_start = [](std::vector<std::string> &term) -> bool {
            string &label = term[2];

            return (label[0] == 'B' || label[0] == 'O' || label[0] == 'S');
        };

        RawSequence::iterator start_it = raw_sequence.begin();
        while (start_it != raw_sequence.end()) {
            RawSequence::iterator next_start_it = find_if(start_it + 1, raw_sequence.end(), is_chunk_start);

            RawSequence::iterator end_it = next_start_it - 1;

            auto len = distance(start_it, end_it);
            if (len == 0) {
                string label = (*start_it)[2];

                if (label[0] == 'S' || label[0] == 'B') {
                    (*start_it)[2] = "B" + label.substr(1);
                }
            } else {
                string label = (*end_it)[2].substr(1);

                (*end_it)[2] = "I" + label;

                (*start_it)[2] = "B" + label;


                if (len >= 2) {
                    for (auto it = start_it + 1; it < end_it; it++) {
                        (*it)[2] = "I" + label;
                    }
                }
            }

            start_it = next_start_it;
        }
    };

    // transform the ETYPE format to the standard BIO format
    // process_sequence();

    for (auto &term : raw_sequence) {
        for (auto i = 0; i < term.size(); i++) {
            os << term[i];
            if (i == term.size() - 1) {
                os << std::endl;
            } else {
                os << " ";
            }
        }
    }

    return os;
}

std::istream& operator>> (std::istream &is, RawSequence &raw_sequence) {
    using namespace std;

    std::string line;

    getline(is, line);
    while (is && !line.empty()) {
        std::istringstream iss(line);

        std::string word, tag, label;
        iss >> word >> tag >> label;
        // std::string word = LabeledTerm::processWord(line);

        std::vector<std::string> term = {word, tag, label};
        // element.push_back(word);
        // element.push_back(tag);
        // element.push_back(label);

        raw_sequence.push_back(term);

        getline(is, line);
    }

    auto process_sequence = [&raw_sequence]() {
        auto is_chunk_start = [](std::vector<std::string> &term) -> bool {
            string &label = term[2];

            return (label[0] == 'B' || label[0] == 'O');
        };

        RawSequence::iterator start_it = raw_sequence.begin();
        while (start_it != raw_sequence.end()) {
            RawSequence::iterator next_start_it = find_if(start_it + 1, raw_sequence.end(), is_chunk_start);

            RawSequence::iterator end_it = next_start_it - 1;

            auto len = distance(start_it, end_it);
            if (len == 0) {
                string label = (*start_it)[2];

                if (label[0] == 'B') {
                    (*start_it)[2] = "S" + label.substr(1);
                }
            } else {
                string label = (*start_it)[2];

                (*start_it)[2] = "B";

                (*end_it)[2] = "E" + label.substr(1);

                if (len >= 2) {
                    for (auto it = start_it + 1; it < end_it; it++) {
                        (*it)[2] = "I";
                    }
                }
            }

            start_it = next_start_it;
        }
    };

    // transform the standard BIO format to the ETYPE format
    process_sequence();

    return is;
}

RawSequence transformEtypeFormat2BIOFormat(RawSequence &raw_sequence) {
    using namespace std;

    auto is_chunk_start = [](std::vector<std::string> &term) -> bool {
        string &label = term[2];

        return (label[0] == 'B' || label[0] == 'O' || label[0] == 'S');
    };

    RawSequence ret_sequence = raw_sequence;

    RawSequence::iterator start_it = ret_sequence.begin();
    while (start_it != ret_sequence.end()) {
        RawSequence::iterator next_start_it = find_if(start_it + 1, ret_sequence.end(), is_chunk_start);

        RawSequence::iterator end_it = next_start_it - 1;

        auto len = distance(start_it, end_it);
        if (len == 0) {
            string label = (*start_it)[2];

            if (label[0] == 'S' || label[0] == 'B') {
                (*start_it)[2] = "B" + label.substr(1);
            }
        } else {
            string label = (*end_it)[2].substr(1);

            (*end_it)[2] = "I" + label;

            (*start_it)[2] = "B" + label;


            if (len >= 2) {
                for (auto it = start_it + 1; it < end_it; it++) {
                    (*it)[2] = "I" + label;
                }
            }
        }

        start_it = next_start_it;
    }

    return ret_sequence;
}

std::vector<std::string> transformEtypeFormat2BIOFormat(std::vector<std::string> &raw_labels) {
    using namespace std;

    auto is_chunk_start = [](std::string &label) -> bool {
        return (label[0] == 'B' || label[0] == 'O' || label[0] == 'S');
    };

    vector<string> ret_labels = raw_labels;

    vector<string>::iterator start_it = ret_labels.begin();
    while (start_it != ret_labels.end()) {
        vector<string>::iterator next_start_it = find_if(start_it + 1, ret_labels.end(), is_chunk_start);

        vector<string>::iterator end_it = next_start_it - 1;

        auto len = distance(start_it, end_it);
        if (len == 0) {
            string label = (*start_it);

            if (label[0] == 'S' || label[0] == 'B') {
                (*start_it) = "B" + label.substr(1);
            }
        } else {
            string label = (*end_it).substr(1);

            (*end_it) = "I" + label;

            (*start_it) = "B" + label;


            if (len >= 2) {
                for (auto it = start_it + 1; it < end_it; it++) {
                    (*it) = "I" + label;
                }
            }
        }

        start_it = next_start_it;
    }

    return ret_labels;

}

std::ostream& operator<< (std::ostream &os, SeqLabelerDataSet &seq_dataset) {
    for (auto &seq : seq_dataset.raw_sequences_) {
        os << seq << std::endl;
    }

    return os;
}

