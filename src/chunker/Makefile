BASE_DIR = ../..
INCLUDE_DIR = $(BASE_DIR)/src/include
CHUNKER_DIR = $(BASE_DIR)/src/chunker

export CC = gcc
export CXX = g++
export NVCC = nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk

export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR)/ -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS = -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)
export NVCCFLAGS =-g -G -O0 -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS)

BIN = $(BASE_DIR)/bin/chunk
OBJ =  Config.o FeatureExtractor.o Beam.o ActionStandardSystem.o
CUOBJ = Chunker.o
CUBIN = 
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

$(BIN) : $(CHUNKER_DIR)/chunk.cpp $(CUOBJ) $(OBJ)

Config.o : $(CHUNKER_DIR)/Config.h $(CHUNKER_DIR)/Config.cpp

chunk.o : $(CHUNKER_DIR)/chunk.cpp $(CHUNKER_DIR)/Chunker.h $(CHUNKER_DIR)/State.h $(CHUNKER_DIR)/Instance.h $(CHUNKER_DIR)/ChunkedSentence.h 

FeatureExtractor.o : $(CHUNKER_DIR)/FeatureExtractor.cpp $(CHUNKER_DIR)/FeatureExtractor.h $(CHUNKER_DIR)/ActionStandardSystem.h

Chunker.o : $(CHUNKER_DIR)/Chunker.cu $(CHUNKER_DIR)/Chunker.h \
	$(CHUNKER_DIR)/State.h $(CHUNKER_DIR)/Instance.h $(CHUNKER_DIR)/ChunkedSentence.h \
	$(CHUNKER_DIR)/FeatureExtractor.h $(CHUNKER_DIR)/Config.h $(CHUNKER_DIR)/Beam.h $(CHUNKER_DIR)/BeamDecodor.h \
	$(CHUNKER_DIR)/Evalb.h \
	$(INCLUDE_DIR)/FeatureEmbedding.h \
	$(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/NNet.h $(INCLUDE_DIR)/Dict.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h

Beam.o : $(CHUNKER_DIR)/Beam.cpp $(CHUNKER_DIR)/Beam.h $(CHUNKER_DIR)/State.h

ActionStandardSystem.o : $(CHUNKER_DIR)/ActionStandardSystem.h $(CHUNKER_DIR)/ActionStandardSystem.cpp $(CHUNKER_DIR)/State.h

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) $(firstword $(filter %.cpp %.c, $^) ) -o $@

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	rm $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
