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

model = smh.listdb_load(Config.MODEL_FILE)
ifs = smh.listdb_load(Config.INVERT_INDEX_FILE)
allObjetsToFileRanked = []
with open(objectsRankingFile, 'rb') as handle:
    allObjetsToFileRanked= pickle.load(handle)

print('Size of all objects discovered: {}'.format(len(allObjetsToFileRanked)))

okAndGoodGroundTruthImages,junkGroundTruthImages = dataLoader.groundTruthLoader(Config.GROUND_TRUTH_PATH)

print('Size of allGroundTruthImages: {}'.format(len(okAndGoodGroundTruthImages)))
print(okAndGoodGroundTruthImages.keys())

allObjetsAP = []
averageAP = 0.0
for key in sorted(okAndGoodGroundTruthImages):
	value = okAndGoodGroundTruthImages[key]
	maxAP = -1
	positiveImages = float(len(value))
	bestIndex = 0
	indexDiscoveredObject = 0

	for objectDiscovered in allObjetsToFileRanked:
		ap = evaluation.computeMyAP(value,junkGroundTruthImages[key],objectDiscovered)
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
	averageAP+=maxAP
print('Average AP: {:.2f}'.format(averageAP/11.0))
		
		




