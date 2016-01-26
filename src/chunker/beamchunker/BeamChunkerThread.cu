/*************************************************************************
	> File Name: BeamChunkerThread.cu
	> Author: cheng chuan
	> Mail: cc.square0@gmail.com 
	> Created Time: Sat 23 Jan 2016 04:26:58 PM CST
 ************************************************************************/
#include "BeamChunkerThread.h"
#include "BeamDecoder.h"

std::ostream& operator<< (std::ostream &os, const GlobalExample &ge) {
    os << ge.instance;
    for (int i = 0; i < ge.goldActs.size(); i++) {
        os << ge.goldActs[i] << " ";
    }
    os << std::endl;

    return os;
}

BeamChunkerThread::BeamChunkerThread(
        const int threadId, 
        const int beamSize, 
        const int miniBatchSize,
        Model<cpu> &paraModel, 
        std::shared_ptr<ActionStandardSystem> transitionSystemPtr, 
        std::shared_ptr<FeatureManager> featureMangerPtr,
        std::shared_ptr<FeatureEmbeddingManager> featureEmbManagerPtr, 
        int longestLen) : 
        m_transSystemPtr(transitionSystemPtr),
        m_featManagerPtr(featureMangerPtr),
        m_featEmbManagerPtr(featureEmbManagerPtr),
        m_nThreadId(threadId), 
        m_nBeamSize(beamSize),
        m_nMiniBatchSize(miniBatchSize)
{
    m_nNumIn = paraModel.Wi2h.shape_[0];
    m_nNumHidden = paraModel.Wi2h.shape_[1];
    m_nNumOut = paraModel.Wh2o.shape_[1];

    InitTensorEngine<gpu>(threadId);

    stream = NewStream<gpu>();

    modelPtr.reset(new Model<gpu>(m_nNumIn, m_nNumHidden, m_nNumOut, paraModel.featTypes, stream, false));
    modelPtr->featEmbs = paraModel.featEmbs;
    modelPtr->featTypes = paraModel.featTypes;

    netPtrs.resize(longestLen + 1);
    for (int i = 0; i < static_cast<int>(netPtrs.size()); i++) {
        netPtrs[i] = new NNet<gpu>(beamSize, m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr.get());
    }
    
    const int nMaxLatticeSize = (beamSize + 1) * longestLen;
    statePtr = new State[nMaxLatticeSize];
    stateIndexPtr = new State*[longestLen + 2];
}

BeamChunkerThread::~BeamChunkerThread() {
    DeleteStream(stream);

    ShutdownTensorEngine<gpu>();

    for (int i = 0; i < static_cast<int>(netPtrs.size()); i++) {
        delete netPtrs[i];
    }

    delete []statePtr;
    delete []stateIndexPtr;
}

void BeamChunkerThread::train(Model<cpu> &paraModel, std::vector<GlobalExample *> &gExamplePtrs, Model<cpu> &cumulatedGrads, double &threadLoss) {
    // copy from the parameter model to current model
    Copy(modelPtr->Wi2h, paraModel.Wi2h, stream);
    Copy(modelPtr->Wh2o, paraModel.Wh2o, stream);
    Copy(modelPtr->hbias, paraModel.hbias, stream);

    Model<gpu> grads(m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr->featTypes, stream);

    for (int i = 0; i < static_cast<int>(gExamplePtrs.size()); i++) {
        GlobalExample *gePtr = gExamplePtrs[i];
        Instance *instPtr = &(gePtr->instance);

        TNNets tnnets(m_nBeamSize, m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr.get(), netPtrs);

        BeamDecoder decoder(instPtr,
                            m_transSystemPtr,
                            m_featManagerPtr,
                            m_featEmbManagerPtr,
                            m_nBeamSize,
                            m_nMiniBatchSize,
                            statePtr,
                            stateIndexPtr,
                            true);

        decoder.decode(tnnets, gePtr);

        tnnets.updateTNNetParas(&grads, decoder, threadLoss);
    }

    // copy grads from current grads to cumulatedGrads
    Copy(cumulatedGrads.Wi2h, grads.Wi2h, stream);
    Copy(cumulatedGrads.Wh2o, grads.Wh2o, stream);
    Copy(cumulatedGrads.hbias, grads.hbias, stream);
    for (int i = 0; i < static_cast<int>(cumulatedGrads.featEmbs.size()); i++) {
        cumulatedGrads.featEmbs[i] = grads.featEmbs[i];
    }
}

void BeamChunkerThread::chunk(const int threads_num, Model<cpu> &paraModel, InstanceSet &devInstances, ChunkedDataSet &labeledSents) {
    Copy(modelPtr->Wi2h, paraModel.Wi2h, stream);
    Copy(modelPtr->Wh2o, paraModel.Wh2o, stream);
    Copy(modelPtr->hbias, paraModel.hbias, stream);

    TNNets tnnets(m_nBeamSize, m_nNumIn, m_nNumHidden, m_nNumOut, modelPtr.get(), false);

    for (unsigned inst = m_nThreadId; inst < static_cast<unsigned>(devInstances.size()); inst += threads_num) {
        LabeledSequence predictSent(devInstances[inst].input);

        BeamDecoder decoder(&(devInstances[inst]), 
                            m_transSystemPtr,
                            m_featManagerPtr,
                            m_featEmbManagerPtr,
                            m_nBeamSize, 
                            m_nMiniBatchSize,
                            statePtr,
                            stateIndexPtr,
                            false);

        decoder.generateLabeledSequence(tnnets, predictSent);

        labeledSents.push_back(predictSent);
    }
}

