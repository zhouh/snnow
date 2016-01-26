/*************************************************************************
	> File Name: GreedyChunkerThread.cu
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 23 Jan 2016 03:30:55 PM CST
 ************************************************************************/
#include "GreedyChunkerThread.h"

GreedyChunkerThread::GreedyChunkerThread(
        const int threadId, 
        const int batchSize, 
        Model<cpu> &paraModel, 
        std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
        std::shared_ptr<FeatureManager> featureMangerPtr,
        std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
        int longestLen) : 
        m_transSystemPtr(transitionSystemPtr),
        m_featManagerPtr(featureMangerPtr),
        m_featEmbManagerPtr(featureEmbManagerPtr),
        m_nThreadId(threadId), 
        m_nBatchSize(batchSize)
{
    m_nNumIn = paraModel.Wi2h.shape_[0];
    m_nNumHidden = paraModel.Wi2h.shape_[1];
    m_nNumOut = paraModel.Wh2o.shape_[1];

    InitTensorEngine<gpu>(threadId);

    stream = NewStream<gpu>();

    modelPtr.reset(new Model<gpu>(m_nNumIn, m_nNumHidden, m_nNumOut, paraModel.featTypes, stream, false, m_nThreadId));
    modelPtr->featEmbs = paraModel.featEmbs;
    modelPtr->featTypes = paraModel.featTypes;

    statePtr = new State[longestLen + 1];
}

GreedyChunkerThread::~GreedyChunkerThread() {
    DeleteStream(stream);

    ShutdownTensorEngine<gpu>();

    delete []statePtr;
}

void GreedyChunkerThread::train(Model<cpu> &paraModel, std::vector<Example *> &examplePtrs, const int miniBatchSize, Model<cpu> &cumulatedGrads, int &threadCorrectSize, double &threadObjLoss) {
    // copy from the parameter model to current model
    Copy(modelPtr->Wi2h, paraModel.Wi2h, stream);
    Copy(modelPtr->Wh2o, paraModel.Wh2o, stream);
    Copy(modelPtr->hbias, paraModel.hbias, stream);

    Model<gpu> grads(m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr->featTypes, stream);

    std::shared_ptr<NNet<gpu>> nnet(new NNet<gpu>(m_nBatchSize, m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr.get()));

    std::vector<FeatureVector> featureVectors(m_nBatchSize);
    TensorContainer<cpu, 2, real_t> input(Shape2(m_nBatchSize, m_nNumIn));

    std::vector<std::vector<int>> validActsVec(m_nBatchSize);
    TensorContainer<cpu, 2, real_t> pred(Shape2(m_nBatchSize, m_nNumOut));

    for (unsigned inst = 0; inst < static_cast<int>(examplePtrs.size()); inst += m_nBatchSize) {
        input = 0.0;
        pred  = 0.0;
        for (unsigned insti = 0; (insti < m_nBatchSize) && (inst + insti < static_cast<int>(examplePtrs.size())); insti++) {
            Example *e = examplePtrs[inst + insti];

            featureVectors[insti] = e->features;
            validActsVec[insti] = e->labels;
        }
        m_featEmbManagerPtr->returnInput(featureVectors, modelPtr->featEmbs, input);

        nnet->Forward(input, pred, CConfig::bDropOut);

        for (unsigned insti = 0; (insti < m_nBatchSize) && (inst + insti < static_cast<int>(examplePtrs.size())); insti++) {
            int optAct = -1;
            int goldAct = -1;

            std::vector<int> &validActs = validActsVec[insti];
            for (int i = 0; i < validActs.size(); i++) {
                if (validActs[i] >= 0) {
                    if (optAct == -1 || pred[insti][i] > pred[insti][optAct]){
                        optAct = i;
                    }

                    if (validActs[i] == 1) {
                        goldAct = i;
                    }
                }
            }
            if (optAct == goldAct) {
                threadCorrectSize += 1;
            }

            real_t maxScore = pred[insti][optAct];
            real_t goldScore = pred[insti][goldAct];

            real_t sum = 0.0;
            for (int i = 0; i < validActs.size(); i++) {
                if (validActs[i] >= 0) {
                    pred[insti][i] = std::exp(pred[insti][i] - maxScore);
                    sum += pred[insti][i];
                }
            }

            threadObjLoss += (std::log(sum) - (goldScore - maxScore)) / miniBatchSize;

            for (int i = 0; i < validActs.size(); i++) {
                if (validActs[i] >= 0) {
                    pred[insti][i] = pred[insti][i] / sum;
                } else {
                    pred[insti][i] = 0.0;
                }
            }
            pred[insti][goldAct] -= 1.0;
        }

        pred /= static_cast<real_t>(miniBatchSize);

        nnet->Backprop(pred);
        nnet->SubsideGradsTo(&grads, featureVectors);
    }

    // copy grads from current grads to cumulatedGrads
    Copy(cumulatedGrads.Wi2h, grads.Wi2h, stream);
    Copy(cumulatedGrads.Wh2o, grads.Wh2o, stream);
    Copy(cumulatedGrads.hbias, grads.hbias, stream);
    for (int i = 0; i < static_cast<int>(cumulatedGrads.featEmbs.size()); i++) {
        cumulatedGrads.featEmbs[i] = grads.featEmbs[i];
    }
}

void GreedyChunkerThread::chunk(const int threads_num, Model<cpu> &paraModel, InstanceSet &devInstances, ChunkedDataSet &labeledSents) {
    Copy(modelPtr->Wi2h, paraModel.Wi2h, stream);
    Copy(modelPtr->Wh2o, paraModel.Wh2o, stream);
    Copy(modelPtr->hbias, paraModel.hbias, stream);

    for (unsigned inst = m_nThreadId; inst < static_cast<unsigned>(devInstances.size()); inst += threads_num) {
        LabeledSequence predictSent(devInstances[inst].input);

        State* predState = decode(&(devInstances[inst]));

        m_transSystemPtr->generateOutput(*predState, predictSent);

        labeledSents.push_back(predictSent);
    }
}

State* GreedyChunkerThread::decode(Instance *inst) {
    Model<gpu> &modelParas = *(modelPtr.get());

    State *lattice = statePtr;

    int nSentLen = inst->input.size();
    int nMaxRound = nSentLen;
    ActionStandardSystem &tranSystem = *(m_transSystemPtr.get());
    std::shared_ptr<NNet<XPU>> nnet(new NNet<XPU>(1, m_nNumIn, m_nNumHidden, m_nNumOut, &modelParas));

    State *retval = nullptr;
    for (int i = 0; i < nMaxRound + 1; ++i) {
        lattice[i].sentLength = nSentLen;
    }

    lattice[0].clear();

    TensorContainer<cpu, 2, real_t> input(Shape2(1, m_nNumIn));
    TensorContainer<cpu, 2, real_t> pred(Shape2(1, m_nNumOut));
       
    for (int nRound = 1; nRound <= nMaxRound; nRound++){
        input = 0.0;
        pred = 0.0;

        State *currentState = lattice + nRound - 1;
        State *target = lattice + nRound;

        std::vector<FeatureVector> featureVectors(1);
        // featureVectors[0].clear();
        generateInputBatch(currentState, inst, featureVectors);
        m_featEmbManagerPtr->returnInput(featureVectors, modelParas.featEmbs, input);

        nnet->Forward(input, pred, false);
        
        std::vector<int> validActs;
        tranSystem.generateValidActs(*currentState, validActs);
        // get max-score valid action
        real_t maxScore = 0.0;
        unsigned maxActID = 0;
        
        for (unsigned actID = 0; actID < validActs.size(); ++actID) {
            if (validActs[actID] == -1) {
                continue;
            }

            if (actID == 0 || pred[0][actID] > maxScore) {
                maxScore = pred[0][actID];
                maxActID = actID;
            }
        }

        CScoredTransition trans(currentState, maxActID, currentState->score + maxScore);
        *target = *currentState;
        tranSystem.move(*currentState, *target, trans);
        retval = target;
    }

    return retval;
}

void GreedyChunkerThread::generateInputBatch(State *state, Instance *inst, std::vector<FeatureVector> &featvecs) {
    for (int i = 0; i < featvecs.size(); i++) {
        m_featManagerPtr->extractFeature(*(state + i), *inst, featvecs[i]);
    }
}
