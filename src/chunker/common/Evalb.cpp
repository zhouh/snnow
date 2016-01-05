/*************************************************************************
	> File Name: Evalb.cpp
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Wed 30 Dec 2015 12:13:22 PM CST
 ************************************************************************/
#include "Evalb.h"


std::tuple<double, double, double> Evalb::eval(ChunkedDataSet &predicts, ChunkedDataSet &golds, bool isEvalNP) {
    assert(predicts.size() == golds.size());

    int correctChunk = 0;
    int foundCorrect = 0;
    int foundGuessed = 0;
   
    for (int i = 0; i < predicts.size(); i++){
        LabeledSequence pscs(predicts[i]);
        LabeledSequence gscs(golds[i]);

        pscs.transform2StardandBIOFormat();
        gscs.transform2StardandBIOFormat();

        auto res = eval(pscs, gscs, isEvalNP);

        correctChunk += std::get<0>(res);
        foundCorrect += std::get<1>(res);
        foundGuessed += std::get<2>(res);
    }

    double precision = 0.0;
    double recall = 0.0;
    double FB1 = 0.0;
    if (foundGuessed != 0) {
        precision = 100 * static_cast<double>(correctChunk) / foundGuessed;
    }

    if (foundCorrect != 0) {
        recall = 100 * static_cast<double>(correctChunk) / foundCorrect;
    }

    if (precision + recall != 0.0) {
        FB1 = 2 * precision * recall / (precision + recall);
    }

    return std::make_tuple(precision, recall, FB1);
}


// return: (correct_count, gold_count, predict_count)
std::tuple<int, int, int> Evalb::eval(LabeledSequence &predict, LabeledSequence &gold, bool isEvalNP) {
    assert (predict.size() == gold.size());

    using std::string;

    string boundary = "-X-";        // sentence boundary
    string correct = "";            // current corpus chunk tag (T, O, B)
    string  correctType = "";        // type of current corpus chunk tag (NP, VP, etc.)
    string guessed = "";            // current guessed chunk tag
    string guessedType = "";        // type of current guessed chunk tag
    string lastCorrect = "O";       // previous chunk tag in corpus
    string lastCorrectType = "";    // type of previous chunk tag in corpus
    string lastGuessed = "O";       // previously identified chunk tag
    string lastGuessedType = "";    // type of previously identified chunk tag

    int foundCorrect = 0;        // number of chunks in corpus
    int foundGuessed = 0;        // number of identified chunks
    int correctChunk = 0;        // number of correctly identified chunks

    int foundCorrectNP = 0;      // number of np chunks in corpus
    int foundGuessedNP = 0;      // number of identified np chunks
    int correctChunkNP = 0;      // number of correctly identified np chunks

    bool inCorrect = false;       // currently processed chunk is correct until now

    const std::vector<LabeledTerm> &correctWords = gold.getLabeledTerms();
    const std::vector<LabeledTerm> &guessedWords = predict.getLabeledTerms();

    string word, correctLabel, guessedLabel;
    for (int i = 0; i <= predict.size(); i++) {
        if (i == predict.size()) {
            word = boundary;
            correctLabel = "O";
            guessedLabel = "O";
        } else {
            word = correctWords[i].word;
            correctLabel = correctWords[i].label;
            guessedLabel = guessedWords[i].label;
        }

        auto found = guessedLabel.find_first_of("-");
        if (found == string::npos) {
            guessed = guessedLabel;
            guessedType = "";
        } else {
            guessed = guessedLabel.substr(0, found);
            guessedType = guessedLabel.substr(found + 1, guessedLabel.size() - found);
        }

        found = correctLabel.find_first_of("-");
        if (found == string::npos) {
            correct = correctLabel;
            correctType = "";
        } else {
            correct = correctLabel.substr(0, found);
            correctType = correctLabel.substr(found + 1, correctLabel.size() - found);
        }

        if (inCorrect) {
            bool correctEndOfChunk = endOfChunk(lastCorrect, correct, lastCorrectType, correctType);
            bool guessedEndOfChunk = endOfChunk(lastGuessed, guessed, lastGuessedType, guessedType);

            if (correctEndOfChunk and guessedEndOfChunk and lastGuessedType == lastCorrectType) {
                inCorrect = false;
                correctChunk++;

                if (isEvalNP && lastCorrectType == "NP") {
                    correctChunkNP++;
                }
            } else if (correctEndOfChunk != guessedEndOfChunk or guessedType != correctType) {
                inCorrect = false;
            }
        }

        bool correctStartOfChunk = startOfChunk(lastCorrect, correct, lastCorrectType, correctType);
        bool guessedStartOfChunk = startOfChunk(lastGuessed, guessed, lastGuessedType, guessedType);
        if (correctStartOfChunk and guessedStartOfChunk and guessedType == correctType) {
            inCorrect = true;
        }

        if (correctStartOfChunk) {
            foundCorrect++;

            if (isEvalNP && correctType == "NP") {
                foundCorrectNP++;
            }
        }

        if (guessedStartOfChunk) {
            foundGuessed++;
            if (isEvalNP && guessedType == "NP") {
                foundGuessedNP++;
            }
        }

        lastGuessed = guessed;
        lastCorrect = correct;
        lastGuessedType = guessedType;
        lastCorrectType = correctType;
    }

    if (inCorrect) {
        correctChunk++;

        if (isEvalNP && lastCorrectType == "NP") {
            correctChunkNP++;
        }
    }

    if (isEvalNP) {
        return std::make_tuple(correctChunkNP, foundCorrectNP, foundGuessedNP);
    } else {
        return std::make_tuple(correctChunk, foundCorrect, foundGuessed);
    }
}

// startOfChunk: checks if a chunk started between the previous and current word
// arguments:    previous and current chunk tags, previous and current types
// note:         this code is not capable of handling other chunk representations
//               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
//               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
bool Evalb::startOfChunk(std::string &prevTag, std::string &tag, std::string &prevType, std::string &type) {
    bool chunkStart = false;

    if (prevTag == "B") {
        if (tag == "B") {
            chunkStart = true;
        }
    } else if (prevTag == "I") {
        if (tag == "B") {
            chunkStart = true;
        }
    } else {
        if (tag == "B" || tag == "I") {
            chunkStart = true;
        }
    }

    if (tag != "O" && tag != "." && prevType != type) {
        chunkStart = true;
    }

    return chunkStart;
}

// endOfChunk: checks if a chunk ended between the preivous and current word
// arguments:  previous and current chunk tags, previous and current types
// note:       this code is not capable of handling other chunk representations
//             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
//             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
bool Evalb::endOfChunk(std::string &prevTag, std::string &tag, std::string prevType, std::string &type) {
    bool chunkEnd = false;

    if (prevTag == "B") {
        if (tag == "B" || tag == "O") {
            chunkEnd = true;
        }
    } else if (prevTag == "I") {
        if (tag == "B" || tag == "O") {
            chunkEnd = true;
        }
    }

    if (prevTag != "O" && prevTag != "." && prevType != type) {
        chunkEnd = true;
    }

    return chunkEnd;
}
