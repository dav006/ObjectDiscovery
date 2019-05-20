import os
import csv
import pickle

inputpath = '../../google_landmark_selected_crop/subfolder/'
files = [None]*20000

index=0
print('Reading Files')
for file in os.listdir(inputpath+'.'):
	removedExt = ".".join(file.split(".")[:-1])
	files[index] = removedExt
	index+=1
	print index

print('Checking if exist')
csvfile = open('train.csv', 'r')
csvreader = csv.reader(csvfile)

count=0
groundTruth = {}
for line in csvreader:
        id = line[0]
	count+=1
	print(count)
	if id not in files:
		continue
	
        landmark = line[2]
	if landmark not in groundTruth:
		groundTruth[landmark] = []
	groundTruth[landmark].append(id)        

print(len(groundTruth))

with open('googleGroundTruth_selected.pickle', 'wb') as handle:
    pickle.dump(groundTruth, handle, protocol=pickle.HIGHEST_PROTOCOL)
