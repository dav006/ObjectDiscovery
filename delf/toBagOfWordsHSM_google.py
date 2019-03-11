from sklearn.cluster import MiniBatchKMeans
from delf import feature_io
import os
from config import Config
import numpy as np
import hnswlib

FILES = 498340
stopWords = set()

inputpath = '../../google_landmark_attention/'
# Reiniting, loading the index
p = hnswlib.Index(space='l2', dim=40)
p.load_index("250000/hsm_250000_15iter_google.bin", max_elements = 250000)

#Convert DELF descriptors to visual words for each delf file
print('Convert DELF')
filelist = sorted(os.listdir(inputpath+'.'))
allVisualWords = [None]*498340
count = 0
for entry in filelist:
	_, _, descriptors, _, _ = feature_io.ReadFromFile(inputpath+entry)
        labels, _ = p.knn_query(descriptors, k=1)
	npArray = labels.flatten()
	visualwords = set(npArray)
	allVisualWords[count] = visualwords
	count+=1
	print(count)

print('Get global count')
# Get global count
globalCount = {}
for wordToFreq in allVisualWords:
	for wordId in wordToFreq:
		if wordId not in globalCount:
			globalCount[wordId] = 1
		else:
			globalCount[wordId] += 1

minCount = 498
maxCount = 149502
#print('Min frecuency : {} and Max frecuency : {}'.format(minFreq,maxFreq))
addStop = 0
for key,value in globalCount.items():
	if value < minCount:
		stopWords.add(key)
		addStop+=1
	elif value > maxCount:
		stopWords.add(key)
		addStop+=1
print('Stop words : {}'.format(addStop))

countVis = 0
for wordToFreq in allVisualWords:
	countVis+=len(wordToFreq)
print('Visual words before stop words : {}'.format(countVis))

#Remove stop words
print('Remove stop words')
allWordToFreqNoStop = []
for wordToFreq in allVisualWords:
	allWordToFreqNoStop.append(wordToFreq.difference(stopWords))

countVis = 0
for wordToFreq in allWordToFreqNoStop:
	countVis+=len(wordToFreq)
print('Visual words after stop words : {}'.format(countVis))

#Remove stop words
print('Save binary BoW without stopWords')
outputFile = open(Config.CORPUS_FILE,'w')
for wordToFreq in allWordToFreqNoStop:
	#Write the dictionary in the default format
	outputFile.write(str(len(wordToFreq)))
	for wordId in wordToFreq:
		outputFile.write(' ')
		outputFile.write(str(wordId)+':1')
	outputFile.write('\n')
outputFile.close