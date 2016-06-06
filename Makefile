BASE_DIR = .
INCLUDE_DIR = $(BASE_DIR)/src/include
DEPPARSER_DIR = $(BASE_DIR)/src/depparser
MSHADOW_DIR = $(BASE_DIR)/thirdparty/mshadow

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk

export CFLAGS = -g -std=c++11 -I$(INCLUDE_DIR)/ -I$(MSHADOW_DIR) -fopenmp $(MSHADOW_CFLAGS)
#export CFLAGS = -Wall -O3 -std=c++11 -I$(INCLUDE_DIR)/ -I$(MSHADOW_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS= -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)
export NVCCFLAGS =-g -G -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS)
#export NVCCFLAGS = -G -O3 -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS)

# specify tensor path
BIN = $(BASE_DIR)/bin/parser
OBJ = DepParseFeatureExtractor.o DepArcStandardSystem.o FeatureEmbedding.o 
CUOBJ = DepParser.o 
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

$(BASE_DIR)/bin/parser : $(DEPPARSER_DIR)/parser.cpp $(CUOBJ) $(OBJ)

DepParseFeatureExtractor.o : $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h

DepParser.o : $(DEPPARSER_DIR)/DepParser.cu $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h

DepArcStandardSystem.o :  $(DEPPARSER_DIR)/*.h $(INCLUDE_DIR)/base/*.h

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -std=c++11 $(firstword $(filter %.cpp %.c, $^) ) -o $@ 

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

#$(DEPPARSER_DIR)/Depparser.o :
#	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter $(DEPPARSER_DIR)/%.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~

