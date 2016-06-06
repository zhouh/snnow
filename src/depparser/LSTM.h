//
// Created by zhouh on 16-4-22.
//

#ifndef SNNOW_LSTM_H
#define SNNOW_LSTM_H

#include <assert.h>
#include <iostream>
#include <memory>

#include "mshadow/tensor.h"

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// #define DEBUG

template<typename xpu>
class LSTMParameter {
public:
    Stream<xpu> *stream_;

    // Forget gate weights
    TensorContainer<xpu, 2, real_t> _Wfx;
    TensorContainer<xpu, 2, real_t> _Wfh;
    TensorContainer<xpu, 2, real_t> _Wfc;
    // TensorContainer<xpu, 2, real_t> Wf;
    TensorContainer<xpu, 1, real_t> _biasf;

    // Input gate weights
    TensorContainer<xpu, 2, real_t> _Wix;
    TensorContainer<xpu, 2, real_t> _Wih;
    TensorContainer<xpu, 2, real_t> _Wic;
    // TensorContainer<xpu, 2, real_t> Wi;
    TensorContainer<xpu, 1, real_t> _biasi;

    // Output gate weights
    TensorContainer<xpu, 2, real_t> _Wox;
    TensorContainer<xpu, 2, real_t> _Woh;
    TensorContainer<xpu, 2, real_t> _Woc;
    // TensorContainer<xpu, 2, real_t> Wo;
    TensorContainer<xpu, 1, real_t> _biaso;

    // Cell weights
    TensorContainer<xpu, 2, real_t> _Wcx;
    TensorContainer<xpu, 2, real_t> _Wch;
    TensorContainer<xpu, 1, real_t> _biasc;

    int input_dim;
    int hidden_dim;
public:
    LSTMParameter() {}
    LSTMParameter(Stream<xpu> * stream) {
        this->stream_ = stream;
    }
    ~LSTMParameter() {}

    void setZero() {
        this->_Wfx   = 0.0;
        this->_Wfh   = 0.0;
        this->_Wfc   = 0.0;
        // this->Wf = 0.0;
        this->_biasf = 0.0;

        this->_Wix   = 0.0;
        this->_Wih   = 0.0;
        this->_Wic   = 0.0;
        // this->Wi = 0.0;
        this->_biasi = 0.0;

        this->_Wox   = 0.0;
        this->_Woh   = 0.0;
        this->_Woc   = 0.0;
        // this->Wo = 0.0;
        this->_biaso = 0.0;

        this->_Wcx   = 0.0;
        this->_Wch   = 0.0;
        this->_biasc = 0.0;
    }

    void set_stream(Stream<xpu> *stream) {
        this->stream_ = stream;
    }

    void zeroInit(const int nInput, const int nHidden) {
        input_dim = nInput;
        hidden_dim = nHidden;

        _Wfx.set_stream(this->stream_);
        _Wfh.set_stream(this->stream_);
        _Wfc.set_stream(this->stream_);
        // Wf.set_stream(this->stream_);
        _biasf.set_stream(this->stream_);

        _Wix.set_stream(this->stream_);
        _Wih.set_stream(this->stream_);
        _Wic.set_stream(this->stream_);
        // Wi.set_stream(this->stream_);
        _biasi.set_stream(this->stream_);

        _Wox.set_stream(this->stream_);
        _Woh.set_stream(this->stream_);
        _Woc.set_stream(this->stream_);
        // Wo.set_stream(this->stream_);
        _biaso.set_stream(this->stream_);

        _Wcx.set_stream(this->stream_);
        _Wch.set_stream(this->stream_);
        _biasc.set_stream(this->stream_);

        _Wfx.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _Wfh.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _Wfc.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        // Wf.Resize(Shape2(xsize, hidden_dim), static_cast<real_t>(0.0));
        _biasf.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _Wix.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _Wih.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _Wic.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        // Wi.Resize(Shape2(xsize, hidden_dim), static_cast<real_t>(0.0));
        _biasi.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _Wox.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _Woh.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _Woc.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        // Wo.Resize(Shape2(xsize, hidden_dim), static_cast<real_t>(0.0));
        _biaso.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _Wcx.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _Wch.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _biasc.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));
    }

    void randomInit(const int nInput, const int nHidden, Random<xpu, real_t> &rnd) {
        zeroInit(nInput, nHidden);

        real_t bound = sqrt(6.0 / (input_dim + 3 * hidden_dim + 1));

        rnd.SampleUniform(&_Wfx, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Wfh, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Wfc, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_biasf, -1.0 * bound, 1.0 * bound);

        rnd.SampleUniform(&_Wix, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Wih, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Wic, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_biasi, -1.0 * bound, 1.0 * bound);

        rnd.SampleUniform(&_Wox, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Woh, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_Woc, -1.0 * bound, 1.0 * bound);
        rnd.SampleUniform(&_biaso, -1.0 * bound, 1.0 * bound);

        rnd.SampleGaussian(&_Wcx, -1.0 * bound, 1.0 * bound);
        rnd.SampleGaussian(&_Wch, -1.0 * bound, 1.0 * bound);
        rnd.SampleGaussian(&_biasc, -1.0 * bound, 1.0 * bound);
    }

    void copyParas(LSTMParameter<cpu> &lstm) {
        Copy(this->_Wfx, lstm._Wfx, stream_);
        Copy(this->_Wfh, lstm._Wfh, stream_);
        Copy(this->_Wfc, lstm._Wfc, stream_);
        Copy(this->_biasf, lstm._biasf, stream_);

        Copy(this->_Wix, lstm._Wix, stream_);
        Copy(this->_Wih, lstm._Wih, stream_);
        Copy(this->_Wic, lstm._Wic, stream_);
        Copy(this->_biasi, lstm._biasi, stream_);

        Copy(this->_Wox, lstm._Wox, stream_);
        Copy(this->_Woh, lstm._Woh, stream_);
        Copy(this->_Woc, lstm._Woc, stream_);
        Copy(this->_biaso, lstm._biaso, stream_);

        Copy(this->_Wcx, lstm._Wcx, stream_);
        Copy(this->_Wch, lstm._Wch, stream_);
        Copy(this->_biasc, lstm._biasc, stream_);
    }

    void copyParas(LSTMParameter<gpu> &lstm) {
        Copy(this->_Wfx, lstm._Wfx, lstm.stream_);
        Copy(this->_Wfh, lstm._Wfh, lstm.stream_);
        Copy(this->_Wfc, lstm._Wfc, lstm.stream_);
        Copy(this->_biasf, lstm._biasf, lstm.stream_);

        Copy(this->_Wix, lstm._Wix, lstm.stream_);
        Copy(this->_Wih, lstm._Wih, lstm.stream_);
        Copy(this->_Wic, lstm._Wic, lstm.stream_);
        Copy(this->_biasi, lstm._biasi, lstm.stream_);

        Copy(this->_Wox, lstm._Wox, lstm.stream_);
        Copy(this->_Woh, lstm._Woh, lstm.stream_);
        Copy(this->_Woc, lstm._Woc, lstm.stream_);
        Copy(this->_biaso, lstm._biaso, lstm.stream_);

        Copy(this->_Wcx, lstm._Wcx, lstm.stream_);
        Copy(this->_Wch, lstm._Wch, lstm.stream_);
        Copy(this->_biasc, lstm._biasc, lstm.stream_);
    }

    void print() {
        std::cerr << "_Wfx: " << std::endl;
        print(_Wfx);
        std::cerr << "_Wfh: " << std::endl;
        print(_Wfh);
        std::cerr << "_Wfc: " << std::endl;
        print(_Wfc);
        std::cerr << "_biasf: " << std::endl;
        print(_biasf);

        std::cerr << "_Wix: " << std::endl;
        print(_Wix);
        std::cerr << "_Wih: " << std::endl;
        print(_Wih);
        std::cerr << "_Wic: " << std::endl;
        print(_Wic);
        std::cerr << "_biasi: " << std::endl;
        print(_biasi);

        std::cerr << "_Wox: " << std::endl;
        print(_Wox);
        std::cerr << "_Woh: " << std::endl;
        print(_Woh);
        std::cerr << "_Woc: " << std::endl;
        print(_Woc);
        std::cerr << "_biaso: " << std::endl;
        print(_biaso);

        std::cerr << "_Wcx: " << std::endl;
        print(_Wcx);
        std::cerr << "_Wch: " << std::endl;
        print(_Wch);
        std::cerr << "_biasc: " << std::endl;
        print(_biasc);
    }

    void print(TensorContainer<cpu, 2, real_t> &tensor) {
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << "\t";
            for (int c = 0; c < tensor.shape_[1]; c++) {
                std::cerr << tensor[r][c] << " ";
            }
            std::cerr << std::endl;
        }
    }

    void print(TensorContainer<gpu, 2, real_t> &tensor) {
        TensorContainer<cpu, 2, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }

    void print(TensorContainer<cpu, 1, real_t> &tensor) {
        std::cerr << "\t";
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << tensor[r] << " ";
        }
        std::cerr << std::endl;
    }

    void print(TensorContainer<gpu, 1, real_t> &tensor) {
        TensorContainer<cpu, 1, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }
/*
    bool hasNaN() {
        if (checkNaN(this->_Wfx)) {
            return true;
        }

        if (checkNaN(this->_Wfh)) {
            return true;
        }

        if (checkNaN(this->_Wfc)) {
            return true;
        }

        if (checkNaN(this->_biasf)) {
            return true;
        }

        if (checkNaN(this->_Wix)) {
            return true;
        }

        if (checkNaN(this->_Wih)) {
            return true;
        }

        if (checkNaN(this->_Wic)) {
            return true;
        }

        if (checkNaN(this->_biasi)) {
            return true;
        }

        if (checkNaN(this->_Wox)) {
            return true;
        }

        if (checkNaN(this->_Woh)) {
            return true;
        }

        if (checkNaN(this->_Woc)) {
            return true;
        }

        if (checkNaN(this->_biaso)) {
            return true;
        }

        if (checkNaN(this->_Wcx)) {
            return true;
        }

        if (checkNaN(this->_Wch)) {
            return true;
        }

        if (checkNaN(this->_biasc)) {
            return true;
        }

        return false;
    }

    void checkNaN() {
        assert (!checkNaN(this->_Wfx));
        assert (!checkNaN(this->_Wfh));
        assert (!checkNaN(this->_Wfc));
        assert (!checkNaN(this->_biasf));

        assert (!checkNaN(this->_Wix));
        assert (!checkNaN(this->_Wih));
        assert (!checkNaN(this->_Wic));
        assert (!checkNaN(this->_biasi));

        assert (!checkNaN(this->_Wox));
        assert (!checkNaN(this->_Woh));
        assert (!checkNaN(this->_Woc));
        assert (!checkNaN(this->_biaso));

        assert (!checkNaN(this->_Wcx));
        assert (!checkNaN(this->_Wch));
        assert (!checkNaN(this->_biasc));
    }

    bool checkNaN(TensorContainer<xpu, 2, real_t> &tensor) {
        TensorContainer<cpu, 2, real_t > ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        for (int r = 0; r < ctensor.shape_[0]; r++){
            for (int c = 0; c < ctensor.shape_[1]; c++) {
                if (std::isnan(ctensor[r][c])) {
                    return true;
                }
            }
        }

        return false;
    }

    bool checkNaN(TensorContainer<xpu, 1, real_t> &tensor) {
        TensorContainer<cpu, 1, real_t > ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        for (int r = 0; r < ctensor.shape_[0]; r++){
            if (std::isnan(ctensor[r])) {
                return true;
            }
        }

        return false;
    }
*/

private:
    LSTMParameter(const LSTMParameter<xpu> &) = delete;
    LSTMParameter<xpu>& operator= (const LSTMParameter<xpu> &) = delete;
};

template<typename xpu>
void mergeParas(LSTMParameter<xpu> &srcLSTM, LSTMParameter<xpu> &dstLSTM) {
    dstLSTM._Wfx += srcLSTM._Wfx;
    dstLSTM._Wfh += srcLSTM._Wfh;
    dstLSTM._Wfc += srcLSTM._Wfc;
    dstLSTM._biasf += srcLSTM._biasf;

    dstLSTM._Wix += srcLSTM._Wix;
    dstLSTM._Wih += srcLSTM._Wih;
    dstLSTM._Wic += srcLSTM._Wic;
    dstLSTM._biasi += srcLSTM._biasi;

    dstLSTM._Wox += srcLSTM._Wox;
    dstLSTM._Woh += srcLSTM._Woh;
    dstLSTM._Woc += srcLSTM._Woc;
    dstLSTM._biaso += srcLSTM._biaso;

    dstLSTM._Wcx += srcLSTM._Wcx;
    dstLSTM._Wch += srcLSTM._Wch;
    dstLSTM._biasc += srcLSTM._biasc;
}

template<typename xpu>
void updateParas(LSTMParameter<xpu> &LSTM, LSTMParameter<xpu> &gradLSTM, LSTMParameter<xpu> &adaGradSquareLSTM) {
    gradLSTM._Wfx += CConfig::fRegularizationRate * LSTM._Wfx;
    gradLSTM._Wfh += CConfig::fRegularizationRate * LSTM._Wfh;
    gradLSTM._Wfc += CConfig::fRegularizationRate * LSTM._Wfc;
    gradLSTM._biasf += CConfig::fRegularizationRate * LSTM._biasf;

    adaGradSquareLSTM._Wfx += F<square>(gradLSTM._Wfx);
    adaGradSquareLSTM._Wfh += F<square>(gradLSTM._Wfh);
    adaGradSquareLSTM._Wfc += F<square>(gradLSTM._Wfc);
    adaGradSquareLSTM._biasf += F<square>(gradLSTM._biasf);

    LSTM._Wfx -= CConfig::fBPRate * ( gradLSTM._Wfx / F<mySqrt>(adaGradSquareLSTM._Wfx + CConfig::fAdaEps) );
    LSTM._Wfh -= CConfig::fBPRate * ( gradLSTM._Wfh / F<mySqrt>(adaGradSquareLSTM._Wfh + CConfig::fAdaEps) );
    LSTM._Wfc -= CConfig::fBPRate * ( gradLSTM._Wfc / F<mySqrt>(adaGradSquareLSTM._Wfc + CConfig::fAdaEps) );
    LSTM._biasf -= CConfig::fBPRate * ( gradLSTM._biasf / F<mySqrt>(adaGradSquareLSTM._biasf + CConfig::fAdaEps) );

    gradLSTM._Wix += CConfig::fRegularizationRate * LSTM._Wix;
    gradLSTM._Wih += CConfig::fRegularizationRate * LSTM._Wih;
    gradLSTM._Wic += CConfig::fRegularizationRate * LSTM._Wic;
    gradLSTM._biasi += CConfig::fRegularizationRate * LSTM._biasi;

    adaGradSquareLSTM._Wix += F<square>(gradLSTM._Wix);
    adaGradSquareLSTM._Wih += F<square>(gradLSTM._Wih);
    adaGradSquareLSTM._Wic += F<square>(gradLSTM._Wic);
    adaGradSquareLSTM._biasi += F<square>(gradLSTM._biasi);

    LSTM._Wix -= CConfig::fBPRate * ( gradLSTM._Wix / F<mySqrt>(adaGradSquareLSTM._Wix + CConfig::fAdaEps) );
    LSTM._Wih -= CConfig::fBPRate * ( gradLSTM._Wih / F<mySqrt>(adaGradSquareLSTM._Wih + CConfig::fAdaEps) );
    LSTM._Wic -= CConfig::fBPRate * ( gradLSTM._Wic / F<mySqrt>(adaGradSquareLSTM._Wic + CConfig::fAdaEps) );
    LSTM._biasi -= CConfig::fBPRate * ( gradLSTM._biasi / F<mySqrt>(adaGradSquareLSTM._biasi + CConfig::fAdaEps) );

    gradLSTM._Wox += CConfig::fRegularizationRate * LSTM._Wox;
    gradLSTM._Woh += CConfig::fRegularizationRate * LSTM._Woh;
    gradLSTM._Woc += CConfig::fRegularizationRate * LSTM._Woc;
    gradLSTM._biaso += CConfig::fRegularizationRate * LSTM._biaso;

    adaGradSquareLSTM._Wox += F<square>(gradLSTM._Wox);
    adaGradSquareLSTM._Woh += F<square>(gradLSTM._Woh);
    adaGradSquareLSTM._Woc += F<square>(gradLSTM._Woc);
    adaGradSquareLSTM._biaso += F<square>(gradLSTM._biaso);

    LSTM._Wox -= CConfig::fBPRate * ( gradLSTM._Wox / F<mySqrt>(adaGradSquareLSTM._Wox + CConfig::fAdaEps) );
    LSTM._Woh -= CConfig::fBPRate * ( gradLSTM._Woh / F<mySqrt>(adaGradSquareLSTM._Woh + CConfig::fAdaEps) );
    LSTM._Woc -= CConfig::fBPRate * ( gradLSTM._Woc / F<mySqrt>(adaGradSquareLSTM._Woc + CConfig::fAdaEps) );
    LSTM._biaso -= CConfig::fBPRate * ( gradLSTM._biaso / F<mySqrt>(adaGradSquareLSTM._biaso + CConfig::fAdaEps) );

    gradLSTM._Wcx += CConfig::fRegularizationRate * LSTM._Wcx;
    gradLSTM._Wch += CConfig::fRegularizationRate * LSTM._Wch;
    gradLSTM._biasc += CConfig::fRegularizationRate * LSTM._biasc;

    adaGradSquareLSTM._Wcx += F<square>(gradLSTM._Wcx);
    adaGradSquareLSTM._Wch += F<square>(gradLSTM._Wch);
    adaGradSquareLSTM._biasc += F<square>(gradLSTM._biasc);

    LSTM._Wcx -= CConfig::fBPRate * ( gradLSTM._Wcx / F<mySqrt>(adaGradSquareLSTM._Wcx + CConfig::fAdaEps) );
    LSTM._Wch -= CConfig::fBPRate * ( gradLSTM._Wch / F<mySqrt>(adaGradSquareLSTM._Wch + CConfig::fAdaEps) );
    LSTM._biasc -= CConfig::fBPRate * ( gradLSTM._biasc / F<mySqrt>(adaGradSquareLSTM._biasc + CConfig::fAdaEps) );
}

template<typename xpu>
class BiDirectionLSTMParameter{
public:
    Stream<xpu> *stream_;

    LSTMParameter<xpu> l2rParas;
    LSTMParameter<xpu> r2lParas;

    int input_dim;
    int hidden_dim;
public:
    BiDirectionLSTMParameter() {}
    BiDirectionLSTMParameter(Stream<xpu> *stream) {
        this->stream_ = stream;
    }

    void set_stream(Stream<xpu> *stream) {
        this->stream_ = stream;

        l2rParas.set_stream(stream);
        r2lParas.set_stream(stream);
    }

    void setZero() {
        l2rParas.setZero();
        r2lParas.setZero();
    }

    void zeroInit(const int nInput, const int nHidden) {
        input_dim = nInput;
        hidden_dim = nHidden;

        l2rParas.zeroInit(input_dim, hidden_dim);
        r2lParas.zeroInit(input_dim, hidden_dim);
    }

    void randomInit(const int nInput, const int nHidden, Random<xpu, real_t> &rnd) {
        input_dim = nInput;
        hidden_dim = nHidden;

        l2rParas.randomInit(input_dim, hidden_dim, rnd);
        r2lParas.randomInit(input_dim, hidden_dim, rnd);
    }

    void copyParas(BiDirectionLSTMParameter<cpu> &biLstm) {
        l2rParas.copyParas(biLstm.l2rParas);
        r2lParas.copyParas(biLstm.r2lParas);
    }

    void copyParas(BiDirectionLSTMParameter<gpu> &biLstm) {
        l2rParas.copyParas(biLstm.l2rParas);
        r2lParas.copyParas(biLstm.r2lParas);
    }

    void print() {
        std::cerr << "l2rParas: " << std::endl;
        l2rParas.print();

        std::cerr << "r2lParas: " << std::endl;
        r2lParas.print();
    }
private:
    BiDirectionLSTMParameter(const BiDirectionLSTMParameter<xpu> &) = delete;
    BiDirectionLSTMParameter<xpu>&operator=(const BiDirectionLSTMParameter<xpu> &) = delete;
};

template<typename xpu>
void mergeParas(BiDirectionLSTMParameter<xpu> &srcBiLSTM, BiDirectionLSTMParameter<xpu> &dstBiLSTM) {
    mergeParas(srcBiLSTM.l2rParas, dstBiLSTM.l2rParas);

    mergeParas(srcBiLSTM.r2lParas, dstBiLSTM.r2lParas);
}

template <typename xpu>
void updateParas(BiDirectionLSTMParameter<xpu> &biLSTM, BiDirectionLSTMParameter<xpu> &biGradLSTM, BiDirectionLSTMParameter<xpu> &adaGradSquareBiLSTM) {
    updateParas(biLSTM.l2rParas, biGradLSTM.l2rParas, adaGradSquareBiLSTM.l2rParas);

    updateParas(biLSTM.r2lParas, biGradLSTM.r2lParas, adaGradSquareBiLSTM.r2lParas);
}

template<typename xpu>
class LSTMNNetUnit {
public:
    LSTMParameter<xpu> *paras;
    Stream<xpu> *stream_;

    int batch_size;
    int input_dim;
    int hidden_dim;

    TensorContainer<xpu, 2, real_t > _xinput;
    TensorContainer<xpu, 2, real_t > _hinput;
    TensorContainer<xpu, 2, real_t > _cinput;

    TensorContainer<xpu, 2, real_t> ft;
    TensorContainer<xpu, 2, real_t> it;
    TensorContainer<xpu, 2, real_t> ot;
    TensorContainer<xpu, 2, real_t> ct;
    TensorContainer<xpu, 2, real_t> yt;
    TensorContainer<xpu, 2, real_t> ct_swung;
    TensorContainer<xpu, 2, real_t> yt_swung;

    TensorContainer<xpu, 2, real_t> lft;
    TensorContainer<xpu, 2, real_t> lit;
    TensorContainer<xpu, 2, real_t> lot;
    TensorContainer<xpu, 2, real_t> lct;
    TensorContainer<xpu, 2, real_t> lyt;
    TensorContainer<xpu, 2, real_t> lct_swung;
    TensorContainer<xpu, 2, real_t> lyt_swung;

    TensorContainer<xpu, 2, real_t> cly;

    // Forget gate weights
    TensorContainer<xpu, 2, real_t> _gradWfx;
    TensorContainer<xpu, 2, real_t> _gradWfh;
    TensorContainer<xpu, 2, real_t> _gradWfc;
    TensorContainer<xpu, 1, real_t> _gradbiasf;

    // Input gate weights
    TensorContainer<xpu, 2, real_t> _gradWix;
    TensorContainer<xpu, 2, real_t> _gradWih;
    TensorContainer<xpu, 2, real_t> _gradWic;
    TensorContainer<xpu, 1, real_t> _gradbiasi;

    // Output gate weights
    TensorContainer<xpu, 2, real_t> _gradWox;
    TensorContainer<xpu, 2, real_t> _gradWoh;
    TensorContainer<xpu, 2, real_t> _gradWoc;
    TensorContainer<xpu, 1, real_t> _gradbiaso;

    // Cell weights
    TensorContainer<xpu, 2, real_t> _gradWcx;
    TensorContainer<xpu, 2, real_t> _gradWch;
    TensorContainer<xpu, 1, real_t> _gradbiasc;

    TensorContainer<xpu, 2, real_t> _gradxinput;
    TensorContainer<xpu, 2, real_t> _gradhinput;
    TensorContainer<xpu, 2, real_t> _gradcinput;

public:
    LSTMNNetUnit(const int nBatchSize, const int nInput, const int nHidden, LSTMParameter<xpu> *paras) {
        batch_size = nBatchSize;
        input_dim = nInput;
        hidden_dim = nHidden;

        this->paras = paras;
        this->stream_ = paras->stream_;

        _xinput.set_stream(stream_);
        _hinput.set_stream(stream_);
        _cinput.set_stream(stream_);

        _xinput.Resize(Shape2(batch_size, input_dim), 0.0);
        _hinput.Resize(Shape2(batch_size, hidden_dim), 0.0);
        _cinput.Resize(Shape2(batch_size, hidden_dim), 0.0);

        ft.set_stream(stream_);
        it.set_stream(stream_);
        ot.set_stream(stream_);
        ct_swung.set_stream(stream_);
        yt_swung.set_stream(stream_);
        yt.set_stream(stream_);
        ct.set_stream(stream_);

        ft.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        it.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        ot.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        ct_swung.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        yt_swung.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        yt.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        ct.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));

    }
    ~LSTMNNetUnit() {}

    void printForward() {
        std::cerr << "_xinput: " << std::endl;
        print(_xinput);
        std::cerr << "_hinput: " << std::endl;
        print(_hinput);
        std::cerr << "_cinput: " << std::endl;
        print(_cinput);

        std::cerr << "ft: " << std::endl;
        print(ft);
        std::cerr << "it: " << std::endl;
        print(it);
        std::cerr << "ot: " << std::endl;
        print(ot);
        std::cerr << "ct: " << std::endl;
        print(ct);
        std::cerr << "yt: " << std::endl;
        print(yt);
        std::cerr << "ct_swung: " << std::endl;
        print(ct_swung);
        std::cerr << "yt_swung: " << std::endl;
        print(yt_swung);
    }

    void printBackward() {
        std::cerr << "lyt_swung: " << std::endl;
        print(lyt_swung);
        std::cerr << "lct_swung: " << std::endl;
        print(lct_swung);
        std::cerr << "lyt: " << std::endl;
        print(lyt);
        std::cerr << "lct: " << std::endl;
        print(lct);
        std::cerr << "lot: " << std::endl;
        print(lot);
        std::cerr << "lit: " << std::endl;
        print(lit);
        std::cerr << "lft: " << std::endl;
        print(lft);


        std::cerr << "_gradxinput: " << std::endl;
        print(_gradxinput);
        std::cerr << "_gradhinput: " << std::endl;
        print(_gradhinput);
        std::cerr << "_gradcinput: " << std::endl;
        print(_gradcinput);
    }

    void printGrads() {
        std::cerr << "_gradWfx: " << std::endl;
        print(_gradWfx);
        std::cerr << "_gradWfh: " << std::endl;
        print(_gradWfh);
        std::cerr << "_gradWfc: " << std::endl;
        print(_gradWfc);
        std::cerr << "_gradbiasf: " << std::endl;
        print(_gradbiasf);

        std::cerr << "_gradWix: " << std::endl;
        print(_gradWix);
        std::cerr << "_gradWih: " << std::endl;
        print(_gradWih);
        std::cerr << "_gradWic: " << std::endl;
        print(_gradWic);
        std::cerr << "_gradbiasi: " << std::endl;
        print(_gradbiasi);

        std::cerr << "_gradWox: " << std::endl;
        print(_gradWox);
        std::cerr << "_gradWoh: " << std::endl;
        print(_gradWoh);
        std::cerr << "_gradWoc: " << std::endl;
        print(_gradWoc);
        std::cerr << "_gradbiaso: " << std::endl;
        print(_gradbiaso);

        std::cerr << "_gradWcx: " << std::endl;
        print(_gradWcx);
        std::cerr << "_gradWch: " << std::endl;
        print(_gradWch);
        std::cerr << "_gradbiasc: " << std::endl;
        print(_gradbiasc);
    }

    void print(TensorContainer<cpu, 2, real_t> &tensor) {
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << "\t";
            for (int c = 0; c < tensor.shape_[1]; c++) {
                std::cerr << tensor[r][c] << " ";
            }
            std::cerr << std::endl;
        }
    }

    void print(TensorContainer<gpu, 2, real_t> &tensor) {
        TensorContainer<cpu, 2, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }

    void print(TensorContainer<cpu, 1, real_t> &tensor) {
        std::cerr << "\t";
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << tensor[r] << " ";
        }
        std::cerr << std::endl;
    }

    void print(TensorContainer<gpu, 1, real_t> &tensor) {
        TensorContainer<cpu, 1, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }

    // input: [xi, hi, ci]
    // output: hidden_out, cell_out
    void Forward(
            TensorContainer<xpu, 2, real_t> &xinput,
            TensorContainer<xpu, 2, real_t> &hinput,
            TensorContainer<xpu, 2, real_t> &cinput,
            TensorContainer<xpu, 2, real_t> &houtput,
            TensorContainer<xpu, 2, real_t> &coutput,
            bool binital = false) {
        index_t batchSize = xinput.size(0);

        Copy(_xinput, xinput, stream_);
        Copy(_hinput, hinput, stream_);
        Copy(_cinput, cinput, stream_);

        // std::cerr << "_xinput: " << std::endl;
        // print(_xinput);
        // std::cerr << "_hinput: " << std::endl;
        // print(_hinput);
        // std::cerr << "_cinput: " << std::endl;
        // print(_cinput);

        if (!binital) {
            // ft = sigma(Wfx * xt + Wfh * ht-1, Wfc * ct-1 + bf)
            triForward(_xinput, _hinput, _cinput,
                       paras->_Wfx, paras->_Wfh, paras->_Wfc, paras->_biasf,
                       ft);
        }

        // std::cerr << "_Wix: " << std::endl;
        // print(paras->_Wix);
        // std::cerr << "_Wih: " << std::endl;
        // print(paras->_Wih);
        // std::cerr << "_Wic: " << std::endl;
        // print(paras->_Wic);

        // it = sigma(Wix * xt + Wih * ht-1, Wic * ct-1 + bf)
        triForward(_xinput, _hinput, _cinput,
                   paras->_Wix, paras->_Wih, paras->_Wic, paras->_biasi,
                   it);
        // std::cerr << "it: " << std::endl;
        // print(it);

        // std::cerr << "_Wcx: " << std::endl;
        // print(paras->_Wcx);
        // std::cerr << "_Wch: " << std::endl;
        // print(paras->_Wch);
        // ct_swung = tanh(Wcx * xt + Wch * ht-1 + bc)
        biForward(_xinput, _hinput,
                  paras->_Wcx, paras->_Wch, paras->_biasc,
                  ct_swung);
        // std::cerr << "ct_swung: " << std::endl;
        // print(ct_swung);

        if (binital) {
            // ct = ft *. ct-1 + it *. ct_swung
            ct = ct_swung * it;
        } else {
            // ct = ft *. ct-1 + it *. ct_swung
            ct = ct_swung * it + _cinput * ft;
        }
        // std::cerr << "ct: " << std::endl;
        // print(ct);

        // ot = sigma(Wox * xt + Woh * ht-1 + Woc * ct + bo)
        triForward(_xinput, _hinput, ct,
                   paras->_Wox, paras->_Woh, paras->_Woc, paras->_biaso,
                   ot);
        // std::cerr << "ot: " << std::endl;
        // print(ot);

        // yt_swung = tanh(ct)
        yt_swung = F<nl_tanh>(ct);
        // std::cerr << "yt_swung: " << std::endl;
        // print(yt_swung);

        // yt = ot *. yt_swung
        yt = yt_swung * ot;
        // std::cerr << "yt: " << std::endl;
        // print(yt);

        Copy(houtput, yt, stream_);
        Copy(coutput, ct, stream_);
    }

    void Backprop(
            TensorContainer<xpu, 2, real_t> &ly,
            TensorContainer<xpu, 2, real_t> &lc,
            TensorContainer<xpu, 2, real_t> &lxin,
            TensorContainer<xpu, 2, real_t> &lhin,
            TensorContainer<xpu, 2, real_t> &lcin,
            bool binital = false) {
        initBackpropVaraiables();

        _gradxinput = 0.0;
        _gradhinput = 0.0;
        _gradcinput = 0.0;

        lyt += ly;
        lct += lc;

        // input: lyt
        // yt = ot *. yt_swung
        lyt_swung = lyt * ot;
        lot = lyt * yt_swung;

        // yt_swung = tanh(ct)
        lct += lyt_swung * F<nl_dtanh>(yt_swung);

        // ot = sigma(Wox * xt + Woh * ht-1 + Woc * ct + bo)
        triBackprop(_xinput, _hinput, ct,
                    ot, lot,
                    paras->_Wox, paras->_Woh, paras->_Woc,
                    _gradxinput, _gradhinput, lct,
                    _gradWox, _gradWoh, _gradWoc, _gradbiaso
        );

        // ct = ft *. ct-1 + it *. ct_swung
        lct_swung = lct * it;
        lit = lct * ct_swung;
        if (!binital) {
            _gradcinput = lct * ft;
            lft = lct * _cinput;
        }

        // ct_swung = tanh(Wcx * xt + Wch * ht-1 + bc)
        biBackprop(_xinput, _hinput,
                   ct_swung, lct_swung,
                   paras->_Wcx, paras->_Wch,
                   _gradxinput, _gradhinput,
                   _gradWcx, _gradWch, _gradbiasc
        );

        // it = sigma(Wix * xt + Wih * ht-1, Wic * ct-1 + bf)
        triBackprop(_xinput, _hinput, _cinput,
                    it, lit,
                    paras->_Wix, paras->_Wih, paras->_Wic,
                    _gradxinput, _gradhinput, _gradcinput,
                    _gradWix, _gradWih, _gradWic, _gradbiasi
        );

        if (!binital) {
            // ft = sigma(Wfx * xt + Wfh * ht-1, Wfc * ct-1 + bf)
            triBackprop(_xinput, _hinput, _cinput,
                        ft, lft,
                        paras->_Wfx, paras->_Wfh, paras->_Wfc,
                        _gradxinput, _gradhinput, _gradcinput,
                        _gradWfx, _gradWfh, _gradWfc, _gradbiasf
            );
        }

        Copy(lxin, _gradxinput, stream_);
        Copy(lhin, _gradhinput, stream_);
        Copy(lcin, _gradcinput, stream_);
    }

    void ResizeForwardTensor(const int batchSize) {
        if (batch_size == batchSize) {
            return ;
        }

        this->batch_size = batchSize;

        Resize(_xinput, batchSize);
        Resize(_hinput, batchSize);
        Resize(_cinput, batchSize);

        Resize(ft, batchSize);
        Resize(it, batchSize);
        Resize(ot, batchSize);
        Resize(ct, batchSize);
        Resize(yt, batchSize);
        Resize(ct_swung, batchSize);
        Resize(yt_swung, batchSize);
    }

    void ResizeBackpropTensor(const int batchSize) {
        if (batch_size == batchSize) {
            return ;
        }

        this->batch_size = batchSize;

        Resize(_gradxinput, batchSize);
        Resize(_gradhinput, batchSize);
        Resize(_gradcinput, batchSize);

        Resize(lft, batchSize);
        Resize(lit, batchSize);
        Resize(lot, batchSize);
        Resize(lct, batchSize);
        Resize(lyt, batchSize);
        Resize(lct_swung, batchSize);
        Resize(lyt_swung, batchSize);

        Resize(cly, batchSize);
    }
/*
    bool hasForwardNaN() {
        if (checkNaN(ft)) {
            return true;
        }

        if (checkNaN(it)) {
            return true;
        }

        if (checkNaN(ot)) {
            return true;
        }

        if (checkNaN(ct)) {
            return true;
        }

        if (checkNaN(yt)) {
            return true;
        }

        if (checkNaN(ct_swung)) {
            return true;
        }

        if (checkNaN(yt_swung)) {
            return true;
        }

        return false;
    }

    void checkForwardNaN() {
        assert (!checkNaN(ft));
        assert (!checkNaN(it));
        assert (!checkNaN(ot));
        assert (!checkNaN(ct));
        assert (!checkNaN(yt));
        assert (!checkNaN(ct_swung));
        assert (!checkNaN(yt_swung));
    }

    bool hasNaN() {
        if (checkNaN(_gradWfx)) {
            return true;
        }
        if (checkNaN(_gradWfh)) {
            return true;
        }
        if (checkNaN(_gradWfc)) {
            return true;
        }
        if (checkNaN(_gradbiasf)) {
            return true;
        }

        if (checkNaN(_gradWix)) {
            return true;
        }
        if (checkNaN(_gradWih)) {
            return true;
        }
        if (checkNaN(_gradWic)) {
            return true;
        }
        if (checkNaN(_gradbiasi)) {
            return true;
        }

        if (checkNaN(_gradWox)) {
            return true;
        }
        if (checkNaN(_gradWoh)) {
            return true;
        }
        if (checkNaN(_gradWoc)) {
            return true;
        }
        if (checkNaN(_gradbiaso)) {
            return true;
        }

        if (checkNaN(_gradWcx)) {
            return true;
        }
        if (checkNaN(_gradWch)) {
            return true;
        }
        if (checkNaN(_gradbiasc)) {
            return true;
        }

        if (checkNaN(_gradxinput)) {
            return true;
        }
        if (checkNaN(_gradhinput)) {
            return true;
        }
        if (checkNaN(_gradcinput)) {
            return true;
        }

        return false;
    }

    void checkNaN() {
        assert (!checkNaN(_gradWfx));
        assert (!checkNaN(_gradWfh));
        assert (!checkNaN(_gradWfc));
        assert (!checkNaN(_gradbiasf));

        assert (!checkNaN(_gradWix));
        assert (!checkNaN(_gradWih));
        assert (!checkNaN(_gradWic));
        assert (!checkNaN(_gradbiasi));

        assert (!checkNaN(_gradWox));
        assert (!checkNaN(_gradWoh));
        assert (!checkNaN(_gradWoc));
        assert (!checkNaN(_gradbiaso));

        assert (!checkNaN(_gradWcx));
        assert (!checkNaN(_gradWch));
        assert (!checkNaN(_gradbiasc));

        assert (!checkNaN(_gradxinput));
        assert (!checkNaN(_gradhinput));
        assert (!checkNaN(_gradcinput));
    }

    bool checkNaN(TensorContainer<xpu, 2, real_t> &tensor) {
        TensorContainer<cpu, 2, real_t > ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        for (int r = 0; r < ctensor.shape_[0]; r++){
            for (int c = 0; c < ctensor.shape_[1]; c++) {
                if (std::isnan(ctensor[r][c])) {
                    return true;
                }
            }
        }

        return false;
    }

    bool checkNaN(TensorContainer<xpu, 1, real_t> &tensor) {
        TensorContainer<cpu, 1, real_t > ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        for (int r = 0; r < ctensor.shape_[0]; r++){
            if (std::isnan(ctensor[r])) {
                return true;
            }
        }

        return false;
    }
*/
private:
    void initBackpropVaraiables() {
        lyt.set_stream(stream_);
        lft.set_stream(stream_);
        lit.set_stream(stream_);
        lot.set_stream(stream_);
        lct.set_stream(stream_);
        lct_swung.set_stream(stream_);
        lyt_swung.set_stream(stream_);

        lyt.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lft.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lit.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lot.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lct.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lct_swung.Resize(Shape2(batch_size, hidden_dim), 0.0);
        lyt_swung.Resize(Shape2(batch_size, hidden_dim), 0.0);

        cly.set_stream(stream_);
        cly.Resize(Shape2(batch_size, hidden_dim), 0.0);

        _gradWfx.set_stream(this->stream_);
        _gradWfh.set_stream(this->stream_);
        _gradWfc.set_stream(this->stream_);
        _gradbiasf.set_stream(this->stream_);

        _gradWix.set_stream(this->stream_);
        _gradWih.set_stream(this->stream_);
        _gradWic.set_stream(this->stream_);
        _gradbiasi.set_stream(this->stream_);

        _gradWox.set_stream(this->stream_);
        _gradWoh.set_stream(this->stream_);
        _gradWoc.set_stream(this->stream_);
        _gradbiaso.set_stream(this->stream_);

        _gradWcx.set_stream(this->stream_);
        _gradWch.set_stream(this->stream_);
        _gradbiasc.set_stream(this->stream_);

        _gradWfx.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWfh.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWfc.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradbiasf.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _gradWix.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWih.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWic.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradbiasi.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _gradWox.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWoh.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWoc.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradbiaso.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _gradWcx.Resize(Shape2(input_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradWch.Resize(Shape2(hidden_dim, hidden_dim), static_cast<real_t>(0.0));
        _gradbiasc.Resize(Shape1(hidden_dim), static_cast<real_t>(0.0));

        _gradxinput.set_stream(stream_);
        _gradhinput.set_stream(stream_);
        _gradcinput.set_stream(stream_);

        _gradxinput.Resize(Shape2(batch_size, input_dim), static_cast<real_t>(0.0));
        _gradhinput.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
        _gradcinput.Resize(Shape2(batch_size, hidden_dim), static_cast<real_t>(0.0));
    }

    void Resize(TensorContainer<xpu, 2, real_t> &container, const int batchSize) {
        const int dimSize = container.shape_[1];

        TensorContainer<xpu, 1, real_t> buff;
        buff.set_stream(stream_);
        buff.Resize(Shape1(dimSize));

        Copy(buff, container[0], stream_);

        container.Resize(Shape2(batchSize, dimSize));
        for (int i = 0; i < batchSize; i++) {
            Copy(container[i], buff, stream_);
        }
    }

    void triForward(
            TensorContainer<xpu, 2, real_t> &xin,
            TensorContainer<xpu, 2, real_t> &hin,
            TensorContainer<xpu, 2, real_t> &cin,
            TensorContainer<xpu, 2, real_t> &Wx,
            TensorContainer<xpu, 2, real_t> &Wh,
            TensorContainer<xpu, 2, real_t> &Wc,
            TensorContainer<xpu, 1, real_t> &bias,
            TensorContainer<xpu, 2, real_t> &out
    ) {
        out  = dot(xin, Wx);
        out += dot(hin, Wh);
        out += dot(cin, Wc);
        out += repmat(bias, batch_size);

        out = F<nl_tanh>(out);
    }

    void triBackprop(
            TensorContainer<xpu, 2, real_t> &xin,
            TensorContainer<xpu, 2, real_t> &hin,
            TensorContainer<xpu, 2, real_t> &cin,
            TensorContainer<xpu, 2, real_t> &output,
            TensorContainer<xpu, 2, real_t> &loutput,
            TensorContainer<xpu, 2, real_t> &Wx,
            TensorContainer<xpu, 2, real_t> &Wh,
            TensorContainer<xpu, 2, real_t> &Wc,
            TensorContainer<xpu, 2, real_t> &lxin,
            TensorContainer<xpu, 2, real_t> &lhin,
            TensorContainer<xpu, 2, real_t> &lcin,
            TensorContainer<xpu, 2, real_t> &gradWx,
            TensorContainer<xpu, 2, real_t> &gradWh,
            TensorContainer<xpu, 2, real_t> &gradWc,
            TensorContainer<xpu, 1, real_t> &gradBias
    ) {
        // loutput: batch_size * hidden_dim
        // cly: batch_size * hidden_dim
        cly = loutput * F<nl_dtanh>(output);

        // gradWx: input_dim * hidden_dim
        // xin: batch_size * input_dim
        gradWx += dot(xin.T(), cly);
        // gradWh: hidden_dim * hidden_dim
        // hin: batch_size * hidden_dim
        gradWh += dot(hin.T(), cly);
        // gradWc: hidden_dim * hidden_dim
        // cin: batch_size * hidden_dim
        gradWc += dot(cin.T(), cly);

        // gradBias: 1 * hidden_dim
        gradBias += sum_rows(cly);

        // xin, lxin: batch_size * input_dim
        // Wx: input_dim * hidden_dim
        lxin += dot(cly, Wx.T());
        // hin, lhin: batch_size * input_dim
        lhin += dot(cly, Wh.T());
        // cin, lcin: batch_size * input_dim
        lcin += dot(cly, Wc.T());
    }

    void biForward(
            TensorContainer<xpu, 2, real_t> &xin,
            TensorContainer<xpu, 2, real_t> &hin,
            TensorContainer<xpu, 2, real_t> &Wx,
            TensorContainer<xpu, 2, real_t> &Wh,
            TensorContainer<xpu, 1, real_t> &bias,
            TensorContainer<xpu, 2, real_t> &out
    ) {
        out  = dot(xin, Wx);
        out += dot(hin, Wh);
        out += repmat(bias, batch_size);

        out = F<nl_tanh>(out);
    }

    void biBackprop(
            TensorContainer<xpu, 2, real_t> &xin,
            TensorContainer<xpu, 2, real_t> &hin,
            TensorContainer<xpu, 2, real_t> &output,
            TensorContainer<xpu, 2, real_t> &loutput,
            TensorContainer<xpu, 2, real_t> &Wx,
            TensorContainer<xpu, 2, real_t> &Wh,
            TensorContainer<xpu, 2, real_t> &lxin,
            TensorContainer<xpu, 2, real_t> &lhin,
            TensorContainer<xpu, 2, real_t> &gradWx,
            TensorContainer<xpu, 2, real_t> &gradWh,
            TensorContainer<xpu, 1, real_t> &gradBias
    ){
        // loutput: batch_size * hidden_dim
        // cly: batch_size * hidden_dim
        cly = loutput * F<nl_dtanh>(output);

        // gradWx: input_dim * hidden_dim
        // xin: batch_size * input_dim
        gradWx += dot(xin.T(), cly);
        // gradWh: hidden_dim * hidden_dim
        // hin: batch_size * hidden_dim
        gradWh += dot(hin.T(), cly);

        // gradBias: 1 * hidden_dim
        gradBias += sum_rows(cly);

        // xin, lxin: batch_size * input_dim
        // Wx: input_dim * hidden_dim
        lxin += dot(cly, Wx.T());
        // hin, lhin: batch_size * input_dim
        lhin += dot(cly, Wh.T());
    }
private:
    LSTMNNetUnit(const LSTMNNetUnit &) = delete;
    LSTMNNetUnit<xpu>&operator=(const LSTMNNetUnit &) = delete;
};

template<typename xpu>
class LSTMNNet {
public:
    LSTMParameter<xpu> *paras;
    Stream<xpu> *stream_;

    int batch_size;
    int input_dim;
    int hidden_dim;

    int seq_size;

    std::vector<std::shared_ptr<LSTMNNetUnit<xpu>>> net_unitPtrs;

    bool left2right;

    TensorContainer<xpu, 2, real_t> hnull, cnull;
    TensorContainer<xpu, 2, real_t> hnullloss, cnullloss;
public:
    LSTMNNet(const int nInput, const int nHidden, const int nSeq, LSTMParameter<xpu> *paras, bool bLeft2Right = true) :
            batch_size(1), input_dim(nInput), hidden_dim(nHidden), seq_size(nSeq), left2right(bLeft2Right)
    {
        for (int i = 0; i < nSeq; i++) {
            net_unitPtrs.push_back(std::shared_ptr<LSTMNNetUnit<xpu>>(new LSTMNNetUnit<xpu>(1, nInput, nHidden, paras)));
        }

        this->paras = paras;
        this->stream_ = paras->stream_;

        hnull.set_stream(stream_);
        cnull.set_stream(stream_);
        hnullloss.set_stream(stream_);
        cnullloss.set_stream(stream_);

        hnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        hnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
    }

    LSTMNNet(const int nBatchSize, const int nInput, const int nHidden, const int nSeq, LSTMParameter<xpu> *paras, bool bLeft2Right = true) :
            batch_size(nBatchSize), input_dim(nInput), hidden_dim(nHidden), seq_size(nSeq), left2right(bLeft2Right)
    {
        for (int i = 0; i < nSeq; i++) {
            net_unitPtrs.push_back(std::shared_ptr<LSTMNNetUnit<xpu>>(new LSTMNNetUnit<xpu>(1, nInput, nHidden, paras)));
        }

        this->paras = paras;
        this->stream_ = paras->stream_;

        hnull.set_stream(stream_);
        cnull.set_stream(stream_);
        hnullloss.set_stream(stream_);
        cnullloss.set_stream(stream_);

        hnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        hnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
    }
    ~LSTMNNet() {}

    void Forward(
            std::vector<TensorContainer<cpu, 2, real_t>> &xins,
    std::vector<TensorContainer<cpu, 2, real_t>> &houtputs
    ) {
        // for (int i = 0; i < seq_size; i++) {
        //     houtputs[i] = xins[i];
        // }

        // return ;

        TensorContainer<xpu, 2, real_t> xin;
        xin.set_stream(stream_);
        xin.Resize(Shape2(batch_size, input_dim), 0.0);
        TensorContainer<xpu, 2, real_t> houtput;
        houtput.set_stream(stream_);
        houtput.Resize(Shape2(batch_size, hidden_dim), 0.0);
        TensorContainer<xpu, 2, real_t> coutput;
        coutput.set_stream(stream_);
        coutput.Resize(Shape2(batch_size, hidden_dim), 0.0);

#ifdef DEBUG
        paras->print();
        char ch;
        std::cin >> ch;
#endif
        if (left2right) {
            for (int idx = 0; idx < seq_size; idx++) {
                Copy(xin, xins[idx], stream_);

                // std::cerr << "xin: " << std::endl;
                // print(xin);
                if (idx == 0) {
                    net_unitPtrs[idx]->Forward(xin, hnull, cnull, houtput, coutput, true);
                } else {
                    net_unitPtrs[idx]->Forward(xin, houtput, coutput, houtput, coutput);
                }

                // std::cerr << "houtput: " << std::endl;
                // print(houtput);
                // std::cerr << "coutput: " << std::endl;
                // print(coutput);

                Copy(houtputs[idx], houtput, stream_);

#ifdef DEBUG
                std::cerr << idx << ":" << std::endl;
                net_unitPtrs[idx]->printForward();
                char ch;
                std::cin >> ch;
#endif
                // std::cerr << "houtputs[idx]: " << std::endl;
                // print(houtputs[idx]);

                // char ch;
                // std::cin >> ch;
            }
        } else {
            for (int idx = seq_size - 1; idx >= 0; idx--) {
                Copy(xin, xins[idx], stream_);

                if (idx == seq_size - 1) {
                    net_unitPtrs[idx]->Forward(xin, hnull, cnull, houtput, coutput, true);
                } else {
                    net_unitPtrs[idx]->Forward(xin, houtput, coutput, houtput, coutput);
                }

#ifdef DEBUG
                std::cerr << idx << ":" << std::endl;
                net_unitPtrs[idx]->printForward();
                // char ch;
                // std::cin >> ch;
#endif
                Copy(houtputs[idx], houtput, stream_);
            }
        }
    }

    void print(TensorContainer<cpu, 2, real_t> &tensor) {
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << "\t";
            for (int c = 0; c < tensor.shape_[1]; c++) {
                std::cerr << tensor[r][c] << " ";
            }
            std::cerr << std::endl;
        }
    }

    void print(TensorContainer<gpu, 2, real_t> &tensor) {
        TensorContainer<cpu, 2, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }

    void print(TensorContainer<cpu, 1, real_t> &tensor) {
        std::cerr << "\t";
        for (int r = 0; r < tensor.shape_[0]; r++){
            std::cerr << tensor[r] << " ";
        }
        std::cerr << std::endl;
    }

    void print(TensorContainer<gpu, 1, real_t> &tensor) {
        TensorContainer<cpu, 1, real_t> ctensor(tensor.shape_);
        Copy(ctensor, tensor, tensor.stream_);

        print(ctensor);
    }

    void Backprop(
            std::vector<TensorContainer<cpu, 2, real_t>> &lys,
    std::vector<TensorContainer<cpu, 2, real_t>> &lxins
    ) {
        TensorContainer<xpu, 2, real_t> lh;
        lh.set_stream(stream_);
        lh.Resize(Shape2(batch_size, hidden_dim), 0.0);

        TensorContainer<xpu, 2, real_t> lFh;
        lFh.set_stream(stream_);
        lFh.Resize(Shape2(batch_size, hidden_dim), 0.0);

        TensorContainer<xpu, 2, real_t> lc;
        lc.set_stream(stream_);
        lc.Resize(Shape2(batch_size, hidden_dim), 0.0);

        TensorContainer<xpu, 2, real_t> lFc;
        lFc.set_stream(stream_);
        lFc.Resize(Shape2(batch_size, hidden_dim), 0.0);

        TensorContainer<xpu, 2, real_t> lxin;
        lxin.set_stream(stream_);
        lxin.Resize(Shape2(batch_size,  input_dim), 0.0);

        if (left2right) {
            for (int idx = seq_size - 1; idx >= 0; idx--) {
                Copy(lh, lys[idx], stream_);

#ifdef DEBUG
                std::cerr << idx << ": " << std::endl;
                std::cerr << "lFh" << std::endl;
                print(lFh);
                std::cerr << "lh" << std::endl;
                print(lh);
                std::cerr << "lFc" << std::endl;
                print(lFc);
#endif
                lh += lFh;
                lc  = lFc;

                net_unitPtrs[idx]->Backprop(lh, lc, lxin, lFh, lFc, (idx == 0));

#ifdef  DEBUG
                net_unitPtrs[idx]->printBackward();
                char ch;
                std::cin >> ch;
#endif

                Copy(lxins[idx], lxin, stream_);
            }
        } else {
            for (int idx = 0; idx < seq_size; idx++) {
                Copy(lh, lys[idx], stream_);

#ifdef DEBUG
                std::cerr << "lFh" << std::endl;
                print(lFh);
                std::cerr << "lh" << std::endl;
                print(lh);
                std::cerr << "lFc" << std::endl;
                print(lFc);
#endif
                lh += lFh;
                lc  = lFc;

                net_unitPtrs[idx]->Backprop(lh, lc, lxin, lFh, lFc, (idx == seq_size - 1));

#ifdef  DEBUG
                std::cerr << idx << ": " << std::endl;
                net_unitPtrs[idx]->printBackward();
                char ch;
                std::cin >> ch;
#endif
                Copy(lxins[idx], lxin, stream_);
            }
        }
        // for (int i = 0; i < seq_size; i++) {
        //     lxins[i] = lys[i];
        // }

        // return ;
    }

    void ResizeForwardTensor(const int batchSize) {
        if (batch_size == batchSize) {
            return ;
        }

        batch_size = batchSize;

        for (int i = 0; i < net_unitPtrs.size(); i++) {
            net_unitPtrs[i]->ResizeForwardTensor(batchSize);
        }

        hnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        hnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
    }

    void ResizeBackpropTensor(const int batchSize) {
        if (batch_size == batchSize) {
            return ;
        }

        batch_size = batchSize;

        for (int i = 0; i < net_unitPtrs.size(); i++) {
            net_unitPtrs[i]->ResizeBackpropTensor(batchSize);
        }

        hnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnull.Resize(Shape2(batch_size, hidden_dim), 0.0);
        hnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
        cnullloss.Resize(Shape2(batch_size, hidden_dim), 0.0);
    }

    void SubsideGradsTo(LSTMParameter<xpu> &LSTMParas) {
#ifdef DEBUG
        std::cerr << "in LSTMNNet SubsideGradsTo: " << std::endl;
#endif
        for (int i = 0; i < net_unitPtrs.size(); i++) {
            LSTMNNetUnit<xpu> *unit = net_unitPtrs[i].get();

#ifdef DEBUG
            // if (i == 0) {
                std::cerr << i << ": " << std::endl;
                unit->printGrads();
            char ch;
            std::cin >> ch;
#endif
            // }

            LSTMParas._Wfx += unit->_gradWfx;
            LSTMParas._Wfh += unit->_gradWfh;
            LSTMParas._Wfc += unit->_gradWfc;
            LSTMParas._biasf += unit->_gradbiasf;

            LSTMParas._Wix += unit->_gradWix;
            LSTMParas._Wih += unit->_gradWih;
            LSTMParas._Wic += unit->_gradWic;
            LSTMParas._biasi += unit->_gradbiasi;

            LSTMParas._Wox += unit->_gradWox;
            LSTMParas._Woh += unit->_gradWoh;
            LSTMParas._Woc += unit->_gradWoc;
            LSTMParas._biaso += unit->_gradbiaso;

            LSTMParas._Wcx += unit->_gradWcx;
            LSTMParas._Wch += unit->_gradWch;
            LSTMParas._biasc += unit->_gradbiasc;
        }

        // char ch;
        // std::cin >> ch;
    }
};

#endif //SNNOW_LSTM_H
