# Script: extract.py

# Description: A script to read json outputs from Cuckoo Sandbox, parse them, 
#              extract APIstats, clean the data, and output it to a file 
#              (extractedInfo.json) for easy use.

# Author: Eric Ciccotelli, 2021
# Organization: Manhattan College
# Contact: eciccotelli01@manhattan.edu

# Maintainer: Daniel Simpson, 2023
# Organization: Tennessee Technological University
# Contact: dnsimpson42@tntech.edu

import glob
import os
import json
import pandas as pd
from pandas import ExcelWriter


#Gets the "apistats" of a file and returns a dictionary containing the key:value pairs
#key = api call: value = number of api calls
def getApiStats(fileName):

    try:
        data = json.load(fileName)
        t = data["behavior"]["apistats"]

        apiStatDict = {}
        for x in t:
            for l in t[x]:
                apiName = str(l)
                apiValue = t[x][l]
                if apiName not in apiStatDict.keys():
                    apiStatDict[apiName] = apiValue
                else:
                    apiStatDict[apiName] += apiValue

        return apiStatDict
    except:
        print("Error parsing api stats with file " + str(fileName))
        return -1


#Removes any duplicates from a list
def removeDuplicates(arr):
    return list(dict.fromkeys(arr))


completeCallList = [] #Holds all API calls for every file (each key will be a row in the table created) using a dict to remove duplicates #COLUMNS
allFilesList = [] #Holds all data for each file, each element is a dictionary
malwareList = [] #Holds row that determines if a file is malware (1) or not (0) #LAST COLUMN
count = 0
bcount = 0

f = open('config.json')

config = json.load(f)
print("\nData path:", config)
f.close()

# taken from config.json. Updated for relative path rather than absolute.
path_to_reports = config['path_to_reports']


for file in glob.glob(path_to_reports + "*.json"):
    with open(file) as f:
        returnedDict = getApiStats(f)
        if returnedDict == -1:
            continue

        if "benign" in file.lower():
            bcount+=1
            malwareList.append(0)
        else:
            malwareList.append(1)
        allFilesList.append(returnedDict)
        callList = list(dict.fromkeys(returnedDict))



        for element in callList:
            completeCallList.append(element)

    count+=1


#completeCallList contains all calls from every file (column headers in table)
completeCallList = removeDuplicates(completeCallList)


print("Checking api calls and compiling lists...")

#Goal: For each malware sample, have a row. columns as API calls
finishedRows = []
count = 0
for fileDict in allFilesList:
    
    temp = []
    for apiCall in completeCallList:

        if apiCall in fileDict.keys():
            temp.append(fileDict[apiCall])
        else:
            temp.append(0)

    finishedRows.append(temp)
    count+=1

print("Lists complete.")

extractedData = {
    "completeCallList": completeCallList,
    "allFilesList": allFilesList,
    "finishedRows": finishedRows,
    "malwareList": malwareList
}


with open("extractedInfo.json", "w") as f:
    json.dump(extractedData, f)
print('Operation Complete!\nWrote extracted data to \'extractedInfo.json\'.\nPlease run the next script, \'student_ML_skeleton.py\'.\n')




