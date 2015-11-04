/*************************************************************************
	> File Name: src/depparser/Evalb.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 08/10/15 20:54:52
 ************************************************************************/

#include <iostream>
#include <tr1/unordered_map>
#include <unordered_set>

#include "assert.h"

class Evalb{

public:

    /*
     * return a pair, first is UAS, second is LAS
     */
    static std::pair<double, double> evalb( std::vector<DepTree> & predicts, std::vector<DepTree> & golds ){

        assert( predicts.size() == golds.size() );

        std::unordered_set<std::string> puncs = {"``", "''", ".", ",", ":"  };
        std::pair<double, double> retval;

        int correctHeadNum = 0;
        int correctArcNum = 0;
        int sumArc = 0;

        for( int i = 0; i < predicts.size(); i++ ){

            assert( predicts[ i ].size == golds[ i ].size );

            for( int j = 1; j < predicts[ i ].size; j++ ){

                std::string tag = predicts[i].nodes[j].tag;
                if( puncs.find(tag) != puncs.end() )
                    continue;

                sumArc++;
                if( predicts[i].nodes[j].head == golds[i].nodes[j].head ){
                    correctHeadNum++;
                    if( predicts[i].nodes[j].label == golds[i].nodes[j].label )
                        correctArcNum++;
                }
            }
        }

        retval.first = (double)correctHeadNum / sumArc; //UAS
        retval.second = (double)correctArcNum / sumArc;

        return retval;
    }

};

