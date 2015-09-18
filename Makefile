BASE_DIR = .
BIN_DIR = ./
INCLUDE_DIR = $(BASE_DIR)/src/include
DEPPARSER_DIR = $(BASE_DIR)/src/depparser

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include $(BASE_DIR)/make/config.mk
include $(BASE_DIR)/make/mshadow.mk
export CFLAGS = -Wall -O3 -std=c++11 -I$(INCLUDE_DIR)/ -I$(INCLUDE_DIR)/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS= -lm -L$(CUDA_HOME)/lib64 $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 -std=c++11 --use_fast_math -ccbin $(CXX)  $(MSHADOW_NVCCFLAGS)

# specify tensor path
BIN = $(BASE_DIR)/bin/parser
OBJ = Config.o FeatureExtractor.o Beam.o
CUOBJ = Depparser.o 
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

$(BASE_DIR)/bin/parser : $(DEPPARSER_DIR)/parser.cpp $(CUOBJ) $(OBJ)

Config.o : $(DEPPARSER_DIR)/Config.h $(DEPPARSER_DIR)/Config.cpp

FeatureExtractor.o : $(DEPPARSER_DIR)/DepTree.h $(DEPPARSER_DIR)/FeatureExtractor.h $(DEPPARSER_DIR)/FeatureExtractor.cpp \
	$(DEPPARSER_DIR)/DepAction.h

Depparser.o : $(DEPPARSER_DIR)/Depparser.cu $(DEPPARSER_DIR)/Depparser.h $(DEPPARSER_DIR)/State.h $(DEPPARSER_DIR)/Config.h $(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/NNet.h $(INCLUDE_DIR)/Dict.h $(DEPPARSER_DIR)/GlobalExample.h $(DEPPARSER_DIR)/Example.h $(INCLUDE_DIR)/mshadow/*.h $(INCLUDE_DIR)/mshadow/extension/*.h $(DEPPARSER_DIR)/Beam.h

#NNet.o : $(INCLUDE_DIR)/NNet.h $(INCLUDE_DIR)/NNet.cu $(INCLUDE_DIR)/mshadow/tensor.h $(INCLUDE_DIR)/mshadow/*.h 

Beam.o : $(DEPPARSER_DIR)/Beam.h $(DEPPARSER_DIR)/Beam.cpp

#$(BASE_DIR)/bin/parser : $(CUOBJ) $(OBJ)

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

