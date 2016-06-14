BASE_DIR = .
INCLUDE_DIR = $(BASE_DIR)/src/include
DEPPARSER_DIR = $(BASE_DIR)/src/depparser
MSHADOW_DIR = $(BASE_DIR)/thirdparty/dmlc-mshadow
OBJ_DIR = $(BASE_DIR)/OBJ

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =/usr/local/cuda-7.0/bin/nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk

export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR)/ -I$(MSHADOW_DIR) -fopenmp $(MSHADOW_CFLAGS) -lgflags
#export CFLAGS = -Wall -O3 -std=c++11 -I$(INCLUDE_DIR)/ -I$(MSHADOW_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS= -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS) -lgflags
export NVCCFLAGS =-g -G -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS)
#export NVCCFLAGS = -G -O3 -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS) -lgflags

# specify tensor path
BIN = $(BASE_DIR)/bin/parser
OBJ = $(OBJ_DIR)/depparser/DepArcStandardSystem.o $(OBJ_DIR)/depparser/DepParseFeatureExtractor.o $(OBJ_DIR)/base/Action.o $(OBJ_DIR)/common/FeatureVector.o
CUOBJ = $(OBJ_DIR)/depparser/DepParser.o $(OBJ_DIR)/common/FeatureEmbedding.o
CUBIN =
.PHONY: clean all

#
# DepParser.o

all: directoires $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

$(BASE_DIR)/bin/parser : $(DEPPARSER_DIR)/parser.cpp $(CUOBJ) $(OBJ)

$(OBJ_DIR)/depparser/DepParseFeatureExtractor.o : $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h $(DEPPARSER_DIR)/DepParseFeatureExtractor.cpp

$(OBJ_DIR)/depparser/DepParser.o : $(DEPPARSER_DIR)/DepParser.cu $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h

$(OBJ_DIR)/depparser/DepArcStandardSystem.o :  $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h $(DEPPARSER_DIR)/DepArcStandardSystem.cpp

$(OBJ_DIR)/common/FeatureEmbedding.o : $(INCLUDE_DIR)/FeatureEmbedding.cu $(INCLUDE_DIR)/*.h

$(OBJ_DIR)/base/Action.o : $(INCLUDE_DIR)/base/Action.cpp $(INCLUDE_DIR)/base/*.h

$(OBJ_DIR)/common/FeatureVector.o : $(INCLUDE_DIR)/FeatureVector.cpp $(INCLUDE_DIR)/*.h

directoires:
	mkdir -p $(OBJ_DIR)/common $(OBJ_DIR)/base $(OBJ_DIR)/depparser

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -std=c++11 $(firstword $(filter %.cpp %.c, $^) ) -o $@ 

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

#$(DEPPARSER_DIR)/Depparser.o :
#	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter $(DEPPARSER_DIR)/%.cu, $^)

#$(CUBIN) :
#	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~

