#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <string>
#include <vector>

using std::string;
using std::vector;

class CConfig
{
public:
	static string strEmbeddingPath;
	static string strTrainPath;
	static string strTestPath;
	static string strdevPath;

	static int nRound;
	static int nBatchSize;
	static int nThread;
	static int nLabelNum;
	static int nBeamSize;
	static int nHiddenSize;
    static int nEmbeddingDim;
    static int nFeatureNum;
    static int nEvaluatePerIters;

	static double fBPRate;
	static double fInitRange;
    static double fRegularizationRate;
    static double fDropoutProb;
    static double fAdaEps;
    

	static bool	 bDropOut;

	static vector<int>  vHiddenSize;
public:
	static bool ReadConfig(const char *pszPath);
	static bool SaveConfig(const char *pszPath);
	static bool LoadConfig(const char *pszPath);
};







#endif  /*__CONFIG_H__*/
