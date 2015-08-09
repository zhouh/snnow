#include "Config.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
/* static members of class config */
int		CConfig::nRound		 = -1;
int 	CConfig::nBatchSize = 100;
int		CConfig::nLeft = 3;
int		CConfig::nRight = 3;
int		CConfig::nThread = 3;
int		CConfig::nBatchNum = 100;
int     CConfig::nEmbeddingDim;

bool  CConfig::bSmall    = false;
bool  CConfig::bBPRateLinearDecay = false;
bool	CConfig::bConvert = false;
bool	CConfig::bLoadDBN = false;
bool	CConfig::bWithHidden = false;
bool	CConfig::bDropOut = false;
bool	CConfig::bHiddenOnly = false;
bool  CConfig::bNormalizeEmbedding = false;

double  CConfig::fBPRate = 0.1;
double  CConfig::fInitMom = 0.5;
double  CConfig::fFinalMom = 0.5;

string CConfig::strRBMPrefix("NULL_STR");
string CConfig::strCorpus("NULL_STR");
string CConfig::strEmbeddingPath("NULL_STR");
string CConfig::strTrainPath("NULL_STR");
string CConfig::strDBNPath("NULL_STR");
string CConfig::strTestPath("NULL_STR");
string CConfig::strErrorType("NULL_STR");
string CConfig::strLogPrefix("NULL_STR");

vector<int> CConfig::vHiddenSize;
vector<string> CConfig::vHiddenType;

bool CConfig::SaveConfig(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "w");
	fclose(fp);
	return true;
}


bool CConfig::LoadConfig(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "r");
	fclose(fp);
	return true;
}


bool CConfig::

/*
 * Get the options from the config file
 * Get size=BUF_LEN char or bit from the file stream and every block is a option
 * */
ReadConfig(const char * pszPath)
{
	FILE *fpIn = fopen(pszPath, "r");
	if (fpIn == NULL)
	{
		fprintf(stderr, "Error: Open %s failed\n", pszPath);
		return false;
	}

	const int BUF_LEN = 256;
	char buf[BUF_LEN];
	while (fgets(buf, BUF_LEN, fpIn) != NULL)
	{
		if (buf [0] == ':' && buf [1] == ':')
			continue;
		
		for (int i = strlen(buf) - 1; i >= 0; --i)
			if (buf[i] == '\r' || buf[i] == '\n')
				buf[i] = 0;
			else
				break;
		
		if (strlen(buf) == 0)
			continue;


		char *pKey = strtok(buf, " \r\t\n");
		if (pKey == NULL)
		{
			fprintf(stderr, "Error: config file invalid format %s\n", buf);
			return false;
		}
		
		char *pVal = strtok(NULL, " \r\t\n");
		
		if (pVal == NULL)
		{
			fprintf(stderr, "Error: config file invalid format %s\n", buf);
			return false;
		}

		if (strcmp(pKey, "nBatchNum") == 0)
		{
			CConfig::nBatchNum = atoi(pVal);
			fprintf(stderr, "nBatchNum %d\n", CConfig::nBatchNum);
		}

		else if (strcmp(pKey, "nThread") == 0)
		{
			CConfig::nThread = atoi(pVal);
			fprintf(stderr, "nThread %d\n", CConfig::nThread);
		}
		
		else if (strcmp(pKey, "nBatchSize") == 0)
		{
			CConfig::nBatchSize = atoi(pVal);
			fprintf(stderr, "nBatchSize %d\n", CConfig::nBatchSize);
		}

		else if (strcmp(pKey, "strTrainPath") == 0)
		{
			CConfig::strTrainPath = pVal;
			fprintf(stderr, "train path %s\n", CConfig::strTrainPath.c_str());
		}

		else if (strcmp(pKey, "strRBMPrefix") == 0)
		{
			CConfig::strRBMPrefix = pVal;
			fprintf(stderr, "WRRBM prefix %s\n", CConfig::strRBMPrefix.c_str());
		}

		else if (strcmp(pKey, "strErrorType") == 0)
		{
			CConfig::strErrorType = pVal;
			fprintf(stderr, "Error function type %s\n", CConfig::strErrorType.c_str());
		}

		else if (strcmp(pKey, "bWithHidden") == 0)
		{
			CConfig::bWithHidden = string(pVal) == "true";
			fprintf(stderr, "Feature with rbm hidden state:%d\n", 
					CConfig::bWithHidden);
		}

		else if (strcmp(pKey, "bDropOut") == 0)
		{
			CConfig::bDropOut = string(pVal) == "true";
			fprintf(stderr, "With Dropout %d", CConfig::bDropOut);
		}

		else if (strcmp(pKey, "bHiddenOnly") == 0)
		{
			CConfig::bHiddenOnly = string(pVal) == "true";
			fprintf(stderr, "only using wrrbm hidden states %d", 
					CConfig::bHiddenOnly);
		}

		else if (strcmp(pKey, "strTestPath") == 0)
		{
			CConfig::strTestPath = pVal;
			fprintf(stderr, "Test path %s\n", CConfig::strTestPath.c_str());
		}

		else if (strcmp(pKey, "strDBNPath") == 0)
		{
			CConfig::strDBNPath = pVal;
			fprintf(stderr, "DBN path %s\n", CConfig::strDBNPath.c_str());
		}

		else if (strcmp(pKey, "nRight") == 0)
		{
			CConfig::nRight = atoi(pVal);
			fprintf(stderr, "nRight: %d\n", CConfig::nRight);
		}

		else if (strcmp(pKey, "strLogPrefix") == 0)
		{
			CConfig::strLogPrefix = pVal;
			fprintf(stderr, "strLogPrefix %s\n", CConfig::strLogPrefix.c_str());
		}

		else if (strcmp(pKey, "nLeft") == 0)
		{
			CConfig::nLeft = atoi(pVal);
			fprintf(stderr, "nLeft: %d\n", CConfig::nLeft);
		}

		else if (strcmp(pKey, "nRound") == 0)
			CConfig::nRound = atoi(pVal);

		else if (strcmp(pKey, "bLoadDBN") == 0)
		{
			CConfig::bLoadDBN = string(pVal) == "true";
			fprintf(stderr, "loading dbn: %s\n", 
					CConfig::bLoadDBN ? "true" : "false");
		}
		
        else if (strcmp(pKey, "nEmbeddingDim") == 0){
            CConfig::nEmbeddingDim = atoi(pVal);
            fprintf(stderr, "The embedding dimension of projection layer:%d\n", CConfig::nEmbeddingDim);
        }

		else if (strcmp(pKey, "strEmbeddingPath") == 0)
		{
			CConfig::strEmbeddingPath = pVal;
			fprintf(stderr, "embedding dict: %s", CConfig::strEmbeddingPath.c_str());
		}
		
		else if (strcmp(pKey, "strCorpus") == 0)
			CConfig::strCorpus = pVal;

		else if (strcmp(pKey, "bSmall") == 0)
		{
			CConfig::bSmall = string(pVal) == string("true");
			fprintf(stderr, "training on small data? %s\n", CConfig::bSmall ? "true":"false");
		}
		
		else if (strcmp(pKey, "fBPRate") == 0)
		{
			CConfig::fBPRate = atof(pVal);
			fprintf(stderr, "BP learning rate:%.5f\n", CConfig::fBPRate);
		}

		else if (strcmp(pKey, "fInitMom") == 0)
		{
			CConfig::fInitMom = atof(pVal);
			fprintf(stderr, "Initial momentum:%.3f\n", CConfig::fInitMom);
		}

		else if (strcmp(pKey, "fFinalMom") == 0)
		{
			CConfig::fFinalMom = atof(pVal);
			fprintf(stderr, "final momentum:%.3f\n", CConfig::fFinalMom);
		}
		else if (strcmp(pKey, "bBPRateLinearDecay") == 0)
		{
			CConfig::bBPRateLinearDecay = string(pVal) == string("true");
			fprintf(stderr, "BP learning weight linear decay:%s\n" ,
					CConfig::bBPRateLinearDecay ? "true":"false");
		}

		else if (strcmp(pKey, "HiddenSize") == 0)
		{
			char *pSizes = pVal; 
			vHiddenSize.clear();
			while (pSizes != NULL)
			{
				vHiddenSize.push_back(atoi(pSizes));
				pSizes = strtok(NULL, " \t\r\n");
			}
		}

		else if (strcmp(pKey, "HiddenType") == 0)
		{
			char *pType = pVal; 
			vHiddenType.clear();
			while (pType != NULL)
			{
				vHiddenType.push_back(pType);
				pType = strtok(NULL, " \t\r\n");
			}
		}
	}
	if (bLoadDBN == false && vHiddenSize.size() > 0 && 
			vHiddenSize.size() != vHiddenType.size())
	{
		fprintf(stderr, "Error: size number and type number inconsistent\n");
		return false;
	}
	return nRound > 0;// && vecHiddenLayerSizes.size() > 0;
}


string CConfig::
BuildPath()
{
	char buf[65536];
	sprintf(buf, "l%d_r%d_", nLeft, nRight);
	string path(buf);
	if (bLoadDBN == true)
		path += "dbn_";

	if (bLoadDBN == false && vHiddenSize.size() > 0)
	{
		path += "hidden_";
		for (size_t i = 0; i < vHiddenSize.size(); ++i)
		{
			sprintf(buf, "%d_%s.", vHiddenSize[i], vHiddenType[i].c_str());
			path += buf;
		}
	}

	path + "_err"+ CConfig::strErrorType;
	return path;
}



//--------------------------------------------------------------------------------------------
