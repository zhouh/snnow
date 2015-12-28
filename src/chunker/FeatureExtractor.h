/*************************************************************************
	> File Name: Feature.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 03:48:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_FEATUREEXTRACTOR_H_
#define _CHUNKER_FEATUREEXTRACTOR_H_

#include <string>
#include <memory>

#include "FeatureEmbedding.h"
#include "DictManager.h"
#include "State.h"
#include "Instance.h"
#include "FeatureType.h"

class FeatureExtractor {
public:
    const FeatureType featType;
    std::shared_ptr<DictManager> dictManagerPtr;

public:
    FeatureExtractor(FeatureType &fType, std::shared_ptr<DictManager> dManagerPtr) : featType(fType), dictManagerPtr(dManagerPtr)
    {
    }
    virtual ~FeatureExtractor() {}

    virtual std::vector<int> extract(State &state, Instance &inst) = 0;

private:
    FeatureExtractor(const FeatureExtractor &fe) = delete;
    FeatureExtractor& operator= (const FeatureExtractor &fe) = delete;
};

class WordFeatureExtractor : public FeatureExtractor {
public:
    WordFeatureExtractor(FeatureType &fType, std::shared_ptr<DictManager> dManagerPtr, FeatureEmbedding *fEmb) : 
    FeatureExtractor(fType, dManagerPtr, fEmb)
    {

    }
    ~WordFeatureExtractor() {}
    
    std::vector<int> extract(State &state, Instance &inst) {
        std::vector<int> features;

        auto getWordIndex = [&state, &inst, this](int index) -> int {
            if (index < 0 || index >= state.m_nLen) {
                return this->dictManagerPtr->nullIdx;
            }

            return inst.wordCache[index];
        };
        
        int currentIndex = state.m_nIndex + 1;
        int IDIdx = 0;

        features.resize(featType.featSize);

        int neg2UniWord   = getWordIndex(currentIndex - 2);
        int neg1UniWord   = getWordIndex(currentIndex - 1);
        int pos0UniWord   = getWordIndex(currentIndex);
        int pos1UniWord   = getWordIndex(currentIndex + 1);
        int pos2UniWord   = getWordIndex(currentIndex + 2);
        features[IDIdx++] = neg2UniWord;
        features[IDIdx++] = neg1UniWord;
        features[IDIdx++] = pos0UniWord;
        features[IDIdx++] = pos1UniWord;
        features[IDIdx++] = pos2UniWord;

        return features;
    }
};

class CapitalFeatureExtractor : public FeatureExtractor {
public:
    CapitalFeatureExtractor(FeatureType &fType, std::shared_ptr<DictManager> dManagerPtr, FeatureEmbedding *fEmb) : 
        FeatureExtractor(fType, dManagerPtr, fEmb)
    {
    }
    ~CapitalFeatureExtractor() {}
    
    std::vector<int> extract(State &state, Instance &inst) {
        std::vector<int> features;

        auto getCapfeatIndex = [&state, &inst, this](int index) -> int {
            if (index < 0 || index >= state.m_nLen) {
                return this->dictManagerPtr->nullIdx;
            }

            return inst.capfeatCache[index];
        };

        int currentIndex = state.m_nIndex + 1;
        int IDIdx = 0;

        features.resize(featType.featSize);

        int pos0UniCap    = getCapfeatIndex(currentIndex);
        //int neg2UniCap    = getCapfeatIndex(currentIndex - 2);
        //int neg1UniCap    = getCapfeatIndex(currentIndex - 1);
        //int pos1UniCap    = getCapfeatIndex(currentIndex + 1);
        //int pos2UniCap    = getCapfeatIndex(currentIndex + 2);
        features[IDIdx++] = pos0UniCap;
        // features[IDIdx++] = neg2UniCap;
        // features[IDIdx++] = neg1UniCap;
        // features[IDIdx++] = pos1UniCap;
        // features[IDIdx++] = pos2UniCap;

        return features;
    }
};

#endif
