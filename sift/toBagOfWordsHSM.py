from sklearn.cluster import MiniBatchKMeans
import os
from config import Config
import numpy as np
import hnswlib

stopWords = set()

inputpath = '/mnt/data/visual_instance_mining/oxbuild_sift/'
# Reiniting, loading the index
p = hnswlib.Index(space='l2', dim=128)
p.load_index("1000000/hsm_1000000_sift_30iter.bin", max_elements = 1000000)

#Convert DELF descriptors to visual words for each delf file
print('Convert DELF')
filelist = sorted(os.listdir(inputpath+'.'))
allVisualWords = []
count = 0
for entry in filelist:
	# Read features
        fileObj = open(inputpath+entry,'r')
        fileObj.readline()
        lineStrip = fileObj.readline().rstrip()
        m = int(lineStrip)
	allDesc = np.empty((m,128))
	index = 0
        for counter in range(m):
                data = fileObj.readline().rstrip()
                dataSplit = data.split(' ')
                dataSplit = dataSplit[5:]
                desc = map(int,dataSplit)
		allDesc[index:index+1,:]=desc
		index+=1
        labels, _ = p.knn_query(allDesc, k=1)
	npArray = labels.flatten()
	visualwords = set(npArray)
	allVisualWords.append(visualwords)
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

minCount = 5
maxCount = 1519
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
