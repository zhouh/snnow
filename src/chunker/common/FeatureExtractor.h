/*************************************************************************
	> File Name: Feature.h
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 26 Dec 2015 03:48:13 PM CST
 ************************************************************************/
#ifndef _CHUNKER_COMMON_FEATUREEXTRACTOR_H_
#define _CHUNKER_COMMON_FEATUREEXTRACTOR_H_

#include <string>
#include <memory>

#include "Dictionary.h"
#include "State.h"
#include "Instance.h"
#include "FeatureType.h"

class FeatureExtractor {
protected:
    FeatureType featType;
    std::shared_ptr<Dictionary> dictPtr;

public:
    FeatureExtractor(const FeatureType &fType, const std::shared_ptr<Dictionary> &dictPtr) : featType(fType), dictPtr(dictPtr)
    {
    }

    const FeatureType& getFeatureType() const {
        return featType;
    }

    const std::shared_ptr<Dictionary>& getDictPtr() const {
        return dictPtr;
    }

    virtual ~FeatureExtractor() {}

    virtual std::vector<int> extract(const State &state, const Instance &inst) = 0;

    // friend std::ostream& operator<< (std::ostream& os, FeatureExtractor &fe);
    // friend std::istream& operator>> (std::istream& is, FeatureExtractor &fe);
private:
    FeatureExtractor(const FeatureExtractor &fe) = delete;
    FeatureExtractor& operator= (const FeatureExtractor &fe) = delete;
};

class WordFeatureExtractor : public FeatureExtractor {
public:
    WordFeatureExtractor(const FeatureType &fType, const std::shared_ptr<Dictionary> &dictPtr) : 
    FeatureExtractor(fType, dictPtr)
    {

    }
    ~WordFeatureExtractor() {}
    
    std::vector<int> extract(const State &state, const Instance &inst) {
        std::vector<int> features;

        auto getWordIndex = [&state, &inst, this](int index) -> int {
            if (index < 0 || index >= state.sentLength) {
                return this->dictPtr->nullIdx;
            }

            return inst.wordCache[index];
        };
        
        int currentIndex = state.index + 1;
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

        // chunk words
        int neg1StartWord = getWordIndex(state.prevChunkIdx);
        int neg1EndWord   = getWordIndex(state.currChunkIdx - 1);
        int pos0StartWord = getWordIndex(state.currChunkIdx);
        int pos0EndWord   = getWordIndex(state.onGoChunkIdx - 1);
        features[IDIdx++] = neg1StartWord;
        features[IDIdx++] = neg1EndWord;
        features[IDIdx++] = pos0StartWord;
        features[IDIdx++] = pos0EndWord;

        return features;
    }
};

class POSFeatureExtractor : public FeatureExtractor {
public:
    POSFeatureExtractor(const FeatureType &fType, const std::shared_ptr<Dictionary> &dictPtr) :
        FeatureExtractor(fType, dictPtr)
    {
    }

    ~POSFeatureExtractor() {}

    std::vector<int> extract(const State &state, const Instance &inst) {
        std::vector<int> features;

        auto getPOSIndex = [&state, &inst, this](int index) -> int {
            if (index < 0 || index >= state.sentLength) {
                return this->dictPtr->nullIdx;
            }

            return inst.tagCache[index];
        };
        
        int currentIndex = state.index + 1;
        int IDIdx = 0;

        features.resize(featType.featSize);

        int neg2UniPOS    = getPOSIndex(currentIndex - 2);
        int neg1UniPOS    = getPOSIndex(currentIndex - 1);
        int pos0UniPOS    = getPOSIndex(currentIndex);
        int pos1UniPOS    = getPOSIndex(currentIndex + 1);
        int pos2UniPOS    = getPOSIndex(currentIndex + 2);
        features[IDIdx++] = neg2UniPOS;
        features[IDIdx++] = neg1UniPOS;
        features[IDIdx++] = pos0UniPOS;
        features[IDIdx++] = pos1UniPOS;
        features[IDIdx++] = pos2UniPOS;

        // chunk pos features
        int neg1StartPOS  = getPOSIndex(state.prevChunkIdx);
        int neg1EndPOS    = getPOSIndex(state.currChunkIdx - 1);
        int pos0StartPOS  = getPOSIndex(state.currChunkIdx);
        int pos0EndPOS    = getPOSIndex(state.onGoChunkIdx - 1);
        features[IDIdx++] = neg1StartPOS;
        features[IDIdx++] = neg1EndPOS;
        features[IDIdx++] = pos0StartPOS;
        features[IDIdx++] = pos0EndPOS;

        return features;
    }
};

class CapitalFeatureExtractor : public FeatureExtractor {
public:
    CapitalFeatureExtractor(const FeatureType &fType, const std::shared_ptr<Dictionary> &dictPtr) : 
        FeatureExtractor(fType, dictPtr)
    {
    }
    ~CapitalFeatureExtractor() {}
    
    std::vector<int> extract(const State &state, const Instance &inst) {
        std::vector<int> features;

        auto getCapfeatIndex = [&state, &inst, this](int index) -> int {
            if (index < 0 || index >= state.sentLength) {
                return this->dictPtr->nullIdx;
            }

            return inst.capfeatCache[index];
        };

        int currentIndex = state.index + 1;
        int IDIdx = 0;

        features.resize(featType.featSize);

        int neg2UniCap    = getCapfeatIndex(currentIndex - 2);
        int neg1UniCap    = getCapfeatIndex(currentIndex - 1);
        int pos0UniCap    = getCapfeatIndex(currentIndex);
        int pos1UniCap    = getCapfeatIndex(currentIndex + 1);
        int pos2UniCap    = getCapfeatIndex(currentIndex + 2);

        features[IDIdx++] = neg2UniCap;
        features[IDIdx++] = neg1UniCap;
        features[IDIdx++] = pos0UniCap;
        features[IDIdx++] = pos1UniCap;
        features[IDIdx++] = pos2UniCap;

        return features;
    }
};

class LabelFeatureExtractor : public FeatureExtractor {
public:
    LabelFeatureExtractor(const FeatureType &fType, const std::shared_ptr<Dictionary> &dictPtr) :
        FeatureExtractor(fType, dictPtr) 
    {
    }
    ~LabelFeatureExtractor() { }

    std::vector<int> extract(const State &state, const Instance &inst) {
        std::vector<int> features;

        auto getLabelIndex = [&state, &inst,  this](int index) -> int {
            if (index < 0) {
                return this->dictPtr->nullIdx;
            }

            return state.chunkedLabelIds[index];
        };

        int currentIndex = state.index + 1;
        int IDIdx = 0;

        features.resize(featType.featSize);

        int neg2UniLabel  = getLabelIndex(currentIndex - 2);
        int neg1UniLabel  = getLabelIndex(currentIndex - 1);
        // int pos0UniLabel  = getLabelIndex(currentIndex);

        features[IDIdx++] = neg2UniLabel;
        features[IDIdx++] = neg1UniLabel;
        // features[IDIdx++] = pos0UniLabel;

        return features;
    }
};

#endif
