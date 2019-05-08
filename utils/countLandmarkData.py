import os
import csv
import pickle
from shutil import copyfile
import operator
import sys

#inputpath = '../../google_landmark_attention/'
inputpath = '../data/google-landmark-dataset/'
files = [None]*1196689

index=0
print('Reading Files')
for file in os.listdir(inputpath+'.'):
	removedExt = ".".join(file.split(".")[:-1])
	files[index] = removedExt
	index+=1

print('Checking if exist')
csvfile = open('train.csv', 'r')
csvreader = csv.reader(csvfile)

count=0
groundTruth = {}
groundTruthCount = {}
for line in csvreader:
        id = line[0]
	if id not in files:
		continue
	
        landmark = line[2]
	if landmark not in groundTruth:
		groundTruth[landmark] = []
		groundTruthCount[landmark] = 0
	groundTruth[landmark].append(id)
	groundTruthCount[landmark] +=1

        count+=1
        print(count)
	sys.stdout.flush()
print(len(groundTruth))
sys.stdout.flush()

with open('groundTruthLand.pickle', 'wb') as handle:
    pickle.dump(groundTruth, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('groundTruthLandCount.pickle', 'wb') as handle:
    pickle.dump(groundTruthCount, handle, protocol=pickle.HIGHEST_PROTOCOL)
