import os
import csv
import pickle
from shutil import copyfile
import operator
import sys

filespath = ['../../annotations_landmarks/annotation_clean_train.txt','../../annotations_landmarks/annotation_clean_val.txt']
datapath = ['../../annotations_landmarks_clean_train/','../../annotations_landmarks_clean_validate/'] 
files = [None]*28360

index=0
print('Reading Files')
for datapathVal in datapath:
	for file in os.listdir(datapathVal+'.'):
		files[index] = file
		index+=1

groundTruth = {}
groundTruthCount = {}
print('Reading labels')
for filepath in filespath:
	csvfile = open(filepath, 'r')
	csvreader = csv.reader(csvfile,delimiter=' ')
	for line in csvreader:
		url=line[0]
		landmark = line[1]
		id = url.split('/')[-1]
		if id not in files:
			continue 
	        if landmark not in groundTruth:
                	groundTruth[landmark] = []
			groundTruthCount[landmark] = 0
			
                groundTruth[landmark].append(id)
		groundTruthCount[landmark] +=1

print(len(groundTruth))
print(groundTruthCount)
