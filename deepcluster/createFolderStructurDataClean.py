import os
import csv
import pickle
from shutil import copyfile
import operator
import sys

filepaths = ['../../annotations_landmarks/annotation_clean_train.txt','../../annotations_landmarks/annotation_clean_val.txt']
datapaths = ['../../annotations_landmarks_clean_train/','../../annotations_landmarks_clean_validate/']

output = '../../annotations_landmarks_clean_struct/'


# Start with train
files = [None]*28358
index=0
print('Reading Files')
for datapath in datapaths:
	for file in os.listdir(datapath+'.'):
		files[index] = file
		index+=1

groundTruth = {}
groundTruthCount = {}
print('Reading labels')
for filepath in filepaths:
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

for landmark in groundTruth:
	folder = output+landmark
	if not os.path.exists(folder):
    		os.makedirs(folder)
	for file in groundTruth[landmark]:
		if os.path.exists(datapaths[0]+file):
			copyfile(datapaths[0]+file, folder+'/'+file)
		else:
			copyfile(datapaths[1]+file, folder+'/'+file)
	

