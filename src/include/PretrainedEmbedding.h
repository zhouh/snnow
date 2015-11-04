/*************************************************************************
	> File Name: src/include/PretrainedEmbedding.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 08/10/15 14:27:31
 ************************************************************************/

#include <iostream>
#include <tr1/unordered_map>
#include <string>

class PretrainedEmbedding{

    std::tr1::unordered_map pretrainWords<std::string, int>;
    std::vector< std::vector<double> > pretrainEmbs;

    /*
     * read the pretrained embedding from the file,
     * returns pretrained word numbers
     */
    int readPretrainEmbeddings( std::string pretrainFile ){

        std::string line;
        std::ifstream in( pretrainFile );
        getline( line, in );

        int index = 0;
        while( getline( line, in ) ){
            std::istringstream iss(line);
            std::string word;
            double d;
            std::vector< double > embedding;

            iss >> word;
            while( iss >> d )
                embedding.push_back( d );

            pretrainFile.push_back( embedding );
            pretrainWords[ word ] = index++;
        }

        return pretrainWords.size();
    }

    int fillPretrainEmbeddings(  );


};
