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

export CFLAGS = -O3 -std=c++11 -I$(INCLUDE_DIR) -I$(CHUNKER_COMMON_DIR) -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
# export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR) -I$(CHUNKER_COMMON_DIR) -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS = -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 -G -std=c++11 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
# export NVCCFLAGS = -g -G -std=c++11 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

GREEDY_CHUNKER_DIR = $(CHUNKER_DIR)/greedychunker
BEAM_CHUNKER_DIR = $(CHUNKER_DIR)/beamchunker

GREEDY_CUOBJ = $(OBJ_DIR)/greedychunker/GreedyChunker.o
BEAM_CUOBJ = $(OBJ_DIR)/beamchunker/BeamChunker.o

GREEDY_BIN = $(BASE_DIR)/bin/greedychunk
BEAM_BIN = $(BASE_DIR)/bin/beamchunk

COMMON_OBJ = $(OBJ_DIR)/common/Config.o $(OBJ_DIR)/common/Evalb.o $(OBJ_DIR)/common/Dictionary.o $(OBJ_DIR)/common/DictManager.o $(OBJ_DIR)/common/FeatureManager.o $(OBJ_DIR)/common/ActionStandardSystem.o
CUDA_COMMON_OBJ = $(OBJ_DIR)/common/FeatureEmbeddingManager.o 
# CUOBJ = $(OBJ_DIR)/Chunker.o
.PHONY: clean all

all: directoires $(GREEDY_BIN)
# all: directoires $(BEAM_BIN)
# all: directoires $(GREEDY_BIN) $(BEAM_BIN)

directoires:
	mkdir -p $(OBJ_DIR)/common $(OBJ_DIR)/greedychunker $(OBJ_DIR)/beamchunker

$(GREEDY_BIN) : $(GREEDY_CHUNKER_DIR)/chunk.cpp $(GREEDY_CUOBJ) $(COMMON_OBJ) $(CUDA_COMMON_OBJ)
$(BEAM_BIN) : $(BEAM_CHUNKER_DIR)/chunk.cpp $(BEAM_CUOBJ) $(COMMON_OBJ) $(CUDA_COMMON_OBJ) $(OBJ_DIR)/beamchunker/Beam.o $(OBJ_DIR)/beamchunker/BeamDecoder.o
# $(BEAM_BIN) : $(BEAM_CHUNKER_DIR)/chunk.cpp $(BEAM_CUOBJ) $(COMMON_OBJ) $(CUDA_COMMON_OBJ) $(OBJ_DIR)/beamchunker/Beam.o 

$(OBJ_DIR)/common/Config.o : $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/Config.cpp

$(OBJ_DIR)/common/Evalb.o : $(CHUNKER_COMMON_DIR)/Evalb.h $(CHUNKER_COMMON_DIR)/Evalb.cpp $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(OBJ_DIR)/common/Dictionary.o : $(CHUNKER_COMMON_DIR)/Dictionary.cpp $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(OBJ_DIR)/common/DictManager.o : $(CHUNKER_COMMON_DIR)/DictManager.cpp $(CHUNKER_COMMON_DIR)/DictManager.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(OBJ_DIR)/common/FeatureManager.o : $(CHUNKER_COMMON_DIR)/FeatureManager.cpp $(CHUNKER_COMMON_DIR)/FeatureManager.h $(CHUNKER_COMMON_DIR)/FeatureExtractor.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/DictManager.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/FeatureType.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(OBJ_DIR)/common/FeatureEmbeddingManager.o : $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.cu $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.h $(CHUNKER_COMMON_DIR)/FeatureEmbedding.h $(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/FeatureType.h $(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/Model.h $(CHUNKER_COMMON_DIR)/NNet.h $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/chunker.h
 
$(OBJ_DIR)/common/ActionStandardSystem.o : $(CHUNKER_COMMON_DIR)/ActionStandardSystem.cpp $(CHUNKER_COMMON_DIR)/ActionStandardSystem.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/LabeledSequence.h

$(OBJ_DIR)/beamchunker/Beam.o : $(BEAM_CHUNKER_DIR)/Beam.cpp $(BEAM_CHUNKER_DIR)/Beam.h
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

COMMON_HEADER = $(CHUNKER_COMMON_DIR)/Config.h $(CHUNKER_COMMON_DIR)/chunker.h \
	$(CHUNKER_COMMON_DIR)/Evalb.h \
	$(CHUNKER_COMMON_DIR)/LabeledSequence.h $(CHUNKER_COMMON_DIR)/Instance.h $(CHUNKER_COMMON_DIR)/State.h $(CHUNKER_COMMON_DIR)/Example.h \
	$(CHUNKER_COMMON_DIR)/Dictionary.h $(CHUNKER_COMMON_DIR)/DictManager.h \
	$(CHUNKER_COMMON_DIR)/FeatureVector.h $(CHUNKER_COMMON_DIR)/FeatureExtractor.h $(CHUNKER_COMMON_DIR)/FeatureManager.h \
	$(CHUNKER_COMMON_DIR)/FeatureEmbedding.h $(CHUNKER_COMMON_DIR)/Model.h $(CHUNKER_COMMON_DIR)/NNet.h $(CHUNKER_COMMON_DIR)/FeatureEmbeddingManager.h \
	$(CHUNKER_COMMON_DIR)/ActionStandardSystem.h \
	$(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h


$(GREEDY_CUOBJ) : $(GREEDY_CHUNKER_DIR)/GreedyChunker.cu $(GREEDY_CHUNKER_DIR)/GreedyChunker.h $(COMMON_HEADER)

$(BEAM_CUOBJ) : $(BEAM_CHUNKER_DIR)/BeamChunker.cu $(BEAM_CHUNKER_DIR)/BeamChunker.h $(COMMON_HEADER) $(BEAM_CHUNKER_DIR)/Beam.h $(BEAM_CHUNKER_DIR)/BeamDecoder.h $(CHUNKER_COMMON_DIR)/chunker.h $(BEAM_CHUNKER_DIR)/TNNets.h

$(OBJ_DIR)/beamchunker/BeamDecoder.o : $(BEAM_CHUNKER_DIR)/BeamDecoder.cu $(BEAM_CHUNKER_DIR)/BeamDecoder.h $(BEAM_CHUNKER_DIR)/TNNets.h
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(GREEDY_BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(BEAM_BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(COMMON_OBJ) :
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

$(CUDA_COMMON_OBJ) :
	$(NVCC) -g -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(GREEDY_CUOBJ) :
	$(NVCC) -g -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(BEAM_CUOBJ) :
	$(NVCC) -g -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	rm $(OBJ_DIR)/*.o $(OBJ_DIR)/greedychunker/*.o $(OBJ_DIR)/beamchunker/*.o $(OBJ_DIR)/common/*.o
