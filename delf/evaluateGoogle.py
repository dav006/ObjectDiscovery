import smh
import pickle
from matplotlib import pyplot
import sklearn.metrics
from config import Config
import sys
sys.path.append('../common/')
import evaluation
import dataLoader

# Folder paths
objectsRankingFile = 'allObjetsRankingFile.pickle'
groundTruthFile = 'googleGroundTruth_selected.pickle'

allObjetsToFileRanked = []
with open(objectsRankingFile, 'rb') as handle:
    allObjetsToFileRanked= pickle.load(handle)

print('Size of all objects discovered: {}'.format(len(allObjetsToFileRanked)))
sys.stdout.flush()

groundTruthImages = {}
with open(groundTruthFile, 'rb') as handle:
    groundTruthImages= pickle.load(handle)

print('Size of allGroundTruthImages: {}'.format(len(groundTruthImages)))
sys.stdout.flush()
allObjetsAP = []
averageAP = 0.0
for key in sorted(groundTruthImages):
	maxAP = -1
        value = groundTruthImages[key]
	bestIndex = 0
	indexDiscoveredObject = 0
	for objectDiscovered in allObjetsToFileRanked:
		ap = evaluation.computeMyAPGoogle(value,objectDiscovered)
		if ap > maxAP:
			maxAP = ap
			bestIndex = indexDiscoveredObject
		indexDiscoveredObject+=1
	'''
	with open(key+'.txt', 'w') as handle:
		for fileOb in allObjetsToFileRanked[bestIndex]:
			handle.write(fileOb+'\n')
	'''
	print('Best AP for {} : {}'.format(key,maxAP))
	sys.stdout.flush()
	'''
	otherMethodAP = evaluation.computeOxfordAP(value,junkGroundTruthImages[key],allObjetsToFileRanked[bestIndex])
	print('Best AP other for {} : {}'.format(key,otherMethodAP))
	'''
	averageAP+=maxAP
	sys.stdout.flush()
print('Average AP: {}'.format(averageAP/20.0))
