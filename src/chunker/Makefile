BASE_DIR = ../..
OBJ_DIR = $(BASE_DIR)/OBJ
INCLUDE_DIR = $(BASE_DIR)/src/include
CHUNKER_DIR = $(BASE_DIR)/src/chunker
CHUNKER_COMMON_DIR = $(CHUNKER_DIR)/common

export CC = gcc
export CXX = g++
export NVCC = nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk

export LDFLAGS = -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)

export CFLAGS = -O3 -std=c++11 -I$(INCLUDE_DIR) -I$(CHUNKER_COMMON_DIR) -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
# export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR) -I$(CHUNKER_COMMON_DIR) -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)

export NVCCFLAGS = -O3 -G -std=c++11 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
# export NVCCFLAGS = -g -G -std=c++11 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

CHUNKER_GREEDY_DIR = $(CHUNKER_DIR)/greedychunker
CHUNKER_BEAM_DIR = $(CHUNKER_DIR)/beamchunker

GREEDY_BIN = $(BASE_DIR)/bin/greedychunk
BEAM_BIN = $(BASE_DIR)/bin/beamchunk

COMMON_OBJ = $(OBJ_DIR)/common/Config.o $(OBJ_DIR)/common/Evalb.o $(OBJ_DIR)/common/Dictionary.o $(OBJ_DIR)/common/DictManager.o $(OBJ_DIR)/common/FeatureManager.o $(OBJ_DIR)/common/ActionStandardSystem.o
COMMON_CUDA_OBJ = $(OBJ_DIR)/common/FeatureEmbeddingManager.o $(OBJ_DIR)/common/FeatureEmbedding.o

GREEDY_CUDA_OBJ = $(OBJ_DIR)/greedychunker/GreedyChunker.o $(OBJ_DIR)/greedychunker/GreedyChunkerThread.o
BEAM_CUDA_OBJ = $(OBJ_DIR)/beamchunker/BeamChunker.o $(OBJ_DIR)/beamchunker/BeamDecoder.o $(OBJ_DIR)/beamchunker/BatchBeamDecoder.o $(OBJ_DIR)/beamchunker/TNNets.o $(OBJ_DIR)/beamchunker/BeamChunkerThread.o

.PHONY: clean all

# all: directoires $(GREEDY_BIN)
all: directoires $(BEAM_BIN)
# all: directoires $(GREEDY_BIN) $(BEAM_BIN)

directoires:
	mkdir -p $(OBJ_DIR)/common $(OBJ_DIR)/greedychunker $(OBJ_DIR)/beamchunker

$(GREEDY_BIN) : $(CHUNKER_GREEDY_DIR)/chunk.cpp $(GREEDY_CUDA_OBJ) $(COMMON_OBJ) $(COMMON_CUDA_OBJ)
$(BEAM_BIN) : $(CHUNKER_BEAM_DIR)/chunk.cpp $(BEAM_CUDA_OBJ) $(COMMON_OBJ) $(COMMON_CUDA_OBJ) $(OBJ_DIR)/beamchunker/Beam.o

$(OBJ_DIR)/common/Config.o : $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/Config.cpp
$(OBJ_DIR)/common/Evalb.o : $(CHUNKER_COMMON_DIR)/Evalb.h $(CHUNKER_COMMON_DIR)/Evalb.cpp $(CHUNKER_COMMON_DIR)/LabeledSequence.h
$(OBJ_DIR)/common/Dictionary.o : $(CHUNKER_COMMON_DIR)/Dictionary.cpp $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h
$(OBJ_DIR)/common/DictManager.o : $(CHUNKER_COMMON_DIR)/DictManager.cpp $(CHUNKER_COMMON_DIR)/DictManager.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h
$(OBJ_DIR)/common/FeatureManager.o : $(CHUNKER_COMMON_DIR)/FeatureManager.cpp $(CHUNKER_COMMON_DIR)/FeatureManager.h $(CHUNKER_COMMON_DIR)/FeatureExtractor.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/DictManager.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/FeatureType.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h
$(OBJ_DIR)/common/ActionStandardSystem.o : $(CHUNKER_COMMON_DIR)/ActionStandardSystem.cpp $(CHUNKER_COMMON_DIR)/ActionStandardSystem.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h


$(OBJ_DIR)/common/FeatureEmbeddingManager.o : $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.cu $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.h $(CHUNKER_COMMON_DIR)/FeatureEmbedding.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/FeatureType.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/Model.h $(CHUNKER_COMMON_DIR)/NNet.h $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/chunker.h
$(OBJ_DIR)/common/FeatureEmbedding.o : $(CHUNKER_COMMON_DIR)/FeatureEmbedding.cu $(CHUNKER_COMMON_DIR)/FeatureEmbedding.h $(CHUNKER_COMMON_DIR)/FeatureType.h

$(OBJ_DIR)/beamchunker/Beam.o : $(CHUNKER_BEAM_DIR)/Beam.cpp $(CHUNKER_BEAM_DIR)/Beam.h
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

$(OBJ_DIR)/greedychunker/GreedyChunkerThread.o : $(CHUNKER_GREEDY_DIR)/GreedyChunkerThread.cu $(CHUNKER_GREEDY_DIR)/GreedyChunkerThread.h $(CHUNKER_COMMON_DIR)/Example.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

COMMON_HEADER = $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/chunker.h \
	$(CHUNKER_COMMON_DIR)/Evalb.h \
	$(CHUNKER_COMMON_DIR)/LabeledSequence.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Example.h \
	$(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/DictManager.h \
	$(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/FeatureExtractor.h $(CHUNKER_COMMON_DIR)/FeatureManager.h \
	$(CHUNKER_COMMON_DIR)/FeatureEmbedding.h $(CHUNKER_COMMON_DIR)/Model.h $(CHUNKER_COMMON_DIR)/NNet.h $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.h \
	$(CHUNKER_COMMON_DIR)/ActionStandardSystem.h \
	$(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h

$(OBJ_DIR)/greedychunker/GreedyChunker.o : $(CHUNKER_GREEDY_DIR)/GreedyChunker.cu $(CHUNKER_GREEDY_DIR)/GreedyChunker.h $(COMMON_HEADER)

$(OBJ_DIR)/beamchunker/BeamChunker.o: $(CHUNKER_BEAM_DIR)/BeamChunker.cu $(CHUNKER_BEAM_DIR)/BeamChunker.h $(COMMON_HEADER) $(CHUNKER_BEAM_DIR)/Beam.h $(CHUNKER_BEAM_DIR)/BeamDecoder.h $(CHUNKER_BEAM_DIR)/BatchBeamDecoder.h $(CHUNKER_COMMON_DIR)/chunker.h $(CHUNKER_BEAM_DIR)/TNNets.h
$(OBJ_DIR)/beamchunker/BeamDecoder.o : $(CHUNKER_BEAM_DIR)/BeamDecoder.cu $(CHUNKER_BEAM_DIR)/BeamDecoder.h $(CHUNKER_BEAM_DIR)/TNNets.h

$(OBJ_DIR)/beamchunker/BatchBeamDecoder.o : $(CHUNKER_BEAM_DIR)/BatchBeamDecoder.cu $(CHUNKER_BEAM_DIR)/BatchBeamDecoder.h $(CHUNKER_BEAM_DIR)/TNNets.h

$(OBJ_DIR)/beamchunker/TNNets.o : $(CHUNKER_BEAM_DIR)/TNNets.h $(CHUNKER_BEAM_DIR)/TNNets.cu

$(OBJ_DIR)/beamchunker/BeamChunkerThread.o : $(CHUNKER_BEAM_DIR)/BeamChunkerThread.cu $(CHUNKER_BEAM_DIR)/BeamChunkerThread.h $(CHUNKER_COMMON_DIR)/Example.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(GREEDY_BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(BEAM_BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(COMMON_OBJ) :
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

$(COMMON_CUDA_OBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(GREEDY_CUDA_OBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(BEAM_CUDA_OBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

clean:
	rm $(OBJ_DIR)/*.o $(OBJ_DIR)/greedychunker/*.o $(OBJ_DIR)/beamchunker/*.o $(OBJ_DIR)/common/*.o
