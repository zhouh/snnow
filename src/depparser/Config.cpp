#include "Config.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
    



/* static members of class config */
string CConfig::strEmbeddingPath("./data/english.em");
string CConfig::strTrainPath("./data/testr.txt");
string CConfig::strTestPath("./data/testr.txt");
string CConfig::strdevPath("./data/devr.txt");

int	CConfig::nRound		 = 10000;
int	CConfig::nBatchSize = 100;
int	CConfig::nThread = 3;
int CConfig::nEmbeddingDim = 50;
int CConfig::nLabelNum = 28;
int CConfig::nBeamSize = 64;
int CConfig::nHiddenSize = 200;
int CConfig::nFeatureNum = 48;

bool CConfig::bDropOut = false;

double CConfig::fBPRate = 0.1;
double CConfig::fInitRange = 0.5;

vector<int> CConfig::vHiddenSize;

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

	}

	return nRound > 0;// && vecHiddenLayerSizes.size() > 0;
}

//--------------------------------------------------------------------------------------------
