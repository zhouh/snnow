BASE_DIR = ../..
OBJ_DIR = $(BASE_DIR)/OBJ
INCLUDE_DIR = $(BASE_DIR)/src/include
CHUNKER_DIR = $(BASE_DIR)/src/chunker

export CC = gcc
export CXX = g++
export NVCC = nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk

export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR)/ -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS = -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -g -G -std=c++11 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

BIN = $(BASE_DIR)/bin/chunk
# OBJ = $(OBJ_DIR)/Config.o $(OBJ_DIR)/Dictionary.o $(OBJ_DIR)/DictManager.o $(OBJ_DIR)/FeatureManager.o $(OBJ_DIR)/Beam.o $(OBJ_DIR)/ActionStandardSystem.o 
# OBJ = $(OBJ_DIR)/Config.o $(OBJ_DIR)/Dictionary.o $(OBJ_DIR)/DictManager.o $(OBJ_DIR)/FeatureManager.o $(OBJ_DIR)/ActionStandardSystem.o 
OBJ = $(OBJ_DIR)/Config.o $(OBJ_DIR)/Dictionary.o $(OBJ_DIR)/DictManager.o $(OBJ_DIR)/FeatureManager.o $(OBJ_DIR)/ActionStandardSystem.o
# CUOBJ = $(OBJ_DIR)/Chunker.o
CUOBJ = $(OBJ_DIR)/GreedyChunker.o
CUBIN = 
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)
# all: $(BIN) $(OBJ)

$(BIN) : $(CHUNKER_DIR)/chunk.cpp $(CUOBJ) $(OBJ)
# $(BIN) : $(OBJ)

# $(OBJ_DIR)/chunk.o : $(CHUNKER_DIR)/chunk.cpp $(CHUNKER_DIR)/Chunker.h $(CHUNKER_DIR)/State.h $(CHUNKER_DIR)/Instance.h $(CHUNKER_DIR)/ChunkedSentence.h 

$(OBJ_DIR)/Config.o : $(CHUNKER_DIR)/Config.h $(CHUNKER_DIR)/Config.cpp

$(OBJ_DIR)/Dictionary.o : $(CHUNKER_DIR)/Dictionary.cpp $(CHUNKER_DIR)/Dictionary.h $(CHUNKER_DIR)/ChunkedSentence.h

$(OBJ_DIR)/DictManager.o : $(CHUNKER_DIR)/DictManager.cpp $(CHUNKER_DIR)/DictManager.h $(CHUNKER_DIR)/Dictionary.h $(CHUNKER_DIR)/ChunkedSentence.h

$(OBJ_DIR)/FeatureManager.o : $(CHUNKER_DIR)/FeatureManager.cpp $(CHUNKER_DIR)/FeatureManager.h
 
# $(OBJ_DIR)/Beam.o : $(CHUNKER_DIR)/Beam.cpp $(CHUNKER_DIR)/Beam.h $(CHUNKER_DIR)/State.h

$(OBJ_DIR)/ActionStandardSystem.o : $(CHUNKER_DIR)/ActionStandardSystem.cpp $(CHUNKER_DIR)/ActionStandardSystem.h $(CHUNKER_DIR)/State.h

# $(OBJ_DIR)/Chunker.o : $(CHUNKER_DIR)/Chunker.cu $(CHUNKER_DIR)/Chunker.h \
# 	$(CHUNKER_DIR)/Config.h \
# 	$(CHUNKER_DIR)/Evalb.h \
# 	$(CHUNKER_DIR)/State.h $(CHUNKER_DIR)/Instance.h $(CHUNKER_DIR)/ChunkedSentence.h $(CHUNKER_DIR)/Example.h \
# 	$(CHUNKER_DIR)/Beam.h $(CHUNKER_DIR)/BeamDecoder.h \
# 	$(CHUNKER_DIR)/Dictionary.h $(CHUNKER_DIR)/DictManager.h \
# 	$(CHUNKER_DIR)/FeatureManager.h $(CHUNKER_DIR)/FeatureVector.h $(CHUNKER_DIR)/FeatureEmbedding.h $(CHUNKER_DIR)/FeatureExtractor.h \
# 	$(CHUNKER_DIR)/FeatureEmbedding.h \
# 	$(INCLUDE_DIR)/NNet.h $(INCLUDE_DIR)/TNNets.h \
# 	$(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/Dict.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h

$(OBJ_DIR)/GreedyChunker.o : $(CHUNKER_DIR)/GreedyChunker.cu $(CHUNKER_DIR)/GreedyChunker.h \
	$(CHUNKER_DIR)/Config.h \
	$(CHUNKER_DIR)/Evalb.h \
	$(CHUNKER_DIR)/State.h $(CHUNKER_DIR)/Instance.h $(CHUNKER_DIR)/ChunkedSentence.h $(CHUNKER_DIR)/Example.h \
	$(CHUNKER_DIR)/Dictionary.h $(CHUNKER_DIR)/DictManager.h \
	$(CHUNKER_DIR)/FeatureManager.h $(CHUNKER_DIR)/FeatureVector.h $(CHUNKER_DIR)/FeatureEmbedding.h $(CHUNKER_DIR)/FeatureExtractor.h \
	$(CHUNKER_DIR)/FeatureEmbedding.h $(CHUNKER_DIR)/Model.h \
	$(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/Dict.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h
 
$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

$(CUOBJ) :
	$(NVCC) -g -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	rm $(OBJ) $(BIN) $(CUBIN) $(CUOBJ)
