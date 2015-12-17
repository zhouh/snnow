/*************************************************************************
	> File Name: Evalb.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Tue 01 Dec 2015 03:44:19 PM CST
 ************************************************************************/
#ifndef _CHUNKER_EVALB_H_
#define _CHUNKER_EVALB_H_

#include <iostream>
#include <string>
#include <assert.h>

#include "ChunkedSentence.h"

class Evalb {
    static std::string BEGIN_I_TYPE;
public:
    // return (precision, recall, FB1)
    static std::tuple<double, double, double> eval(ChunkedDataSet &predicts, ChunkedDataSet &golds, bool isEvalNP = false) {
        // std::cout << "predict.size = " << predicts.size() << "\tgold.size = " << golds.size() << std::endl;
        assert(predicts.size() == golds.size());

        int correctChunk = 0;
        int foundCorrect = 0;
        int foundGuessed = 0;
       
        for (int i = 0; i < predicts.size(); i++){
            // std::cout << "sentence " << (i + 1) << ":" << std::endl;
            // std::vector<std::string> pbefore; 
            // std::vector<std::string> pafter; 
            // std::vector<std::string> gbefore; 
            // std::vector<std::string> gafter; 

            ChunkedSentence pscs(predicts[i]);
            ChunkedSentence gscs(golds[i]);
            // for (int j = 0; j < pscs.m_nLength; j++) {
            //     pbefore.push_back(pscs.m_lChunkedWords[j].label);
            //     gbefore.push_back(gscs.m_lChunkedWords[j].label);
            // }
            convert2StardandBIOFormat(pscs);
            convert2StardandBIOFormat(gscs);
            // for (int j = 0; j < pscs.m_nLength; j++) {
            //     pafter.push_back(pscs.m_lChunkedWords[j].label);
            //     gafter.push_back(gscs.m_lChunkedWords[j].label);
            // }
            // int k = 0;
            // while (k < pbefore.size()) {
            //     std::cout << pbefore[k] << "\t" << gbefore[k] << "\t" << pafter[k] << "\t" << gafter[k] << std::endl;
            //     k++;
            // }

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

    static void convert2StardandBIOFormat(ChunkedSentence &scs) {
        std::string type = BEGIN_I_TYPE;
        for (int i = 0; i < scs.m_nLength; i++) {
            if (scs.m_lChunkedWords[i].label == "I") {
                scs.m_lChunkedWords[i].label = "I-" + type;
            } else if (scs.m_lChunkedWords[i].label[0] == 'B'){
                type = scs.m_lChunkedWords[i].label.substr(2);
            } else {
                type = BEGIN_I_TYPE;
            }
        }
    }

    // return: (correct_count, gold_count, predict_count)
    static std::tuple<int, int, int> eval(ChunkedSentence &predict, ChunkedSentence &gold, bool isEvalNP = false) {
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

        const std::vector<ChunkedWord> &correctWords = gold.getChunkedWords();
        const std::vector<ChunkedWord> &guessedWords = predict.getChunkedWords();

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
private:
    // startOfChunk: checks if a chunk started between the previous and current word
    // arguments:    previous and current chunk tags, previous and current types
    // note:         this code is not capable of handling other chunk representations
    //               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    //               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    static bool startOfChunk(std::string &prevTag, std::string &tag, std::string &prevType, std::string &type) {
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
    static bool endOfChunk(std::string &prevTag, std::string &tag, std::string prevType, std::string &type) {
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
};

std::string Evalb::BEGIN_I_TYPE = "ITYPE";

/*
int main() {
    ChunkedDataSet predicts;
    ChunkedDataSet golds;

    std::string line;

    ChunkedSentence psent;
    ChunkedSentence gsent;
    getline(std::cin, line);
    while (std::cin) {
        if (line.empty()) {
            psent.m_nLength = psent.m_lChunkedWords.size();
            gsent.m_nLength = gsent.m_lChunkedWords.size();

            predicts.push_back(psent);
            golds.push_back(gsent);

            psent.m_lChunkedWords.clear();
            gsent.m_lChunkedWords.clear();
            
            getline(std::cin, line);
            continue;
        }
        std::string word;
        std::string postag;
        std::string correctLabel;
        std::string guessedLabel;

        std::istringstream iss(line);

        iss >> word >> postag >> correctLabel >> guessedLabel;

        ChunkedWord ccw(word, postag, correctLabel);
        gsent.m_lChunkedWords.push_back(ccw);

        ChunkedWord gcw(word, postag, guessedLabel);
        psent.m_lChunkedWords.push_back(gcw);

        getline(std::cin, line);
    }

    auto res = Evalb::eval(predicts, golds, true);

    std::cout << "precision: " << std::get<0>(res) << std::endl;
    std::cout << "recall: " << std::get<1>(res) << std::endl;
    std::cout << "FB1: " << std::get<2>(res) << std::endl;

    return 0;
}
*/

#endif
