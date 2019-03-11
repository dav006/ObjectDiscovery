import smh
import pickle
from matplotlib import pyplot
import sklearn.metrics
from config import Config
import sys
sys.path.append('../common/')
import evaluation
import dataLoader

import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def paralelFunc(value,allObjetsToFileRanked):
        maxAP = -1
	for objectDiscovered in allObjetsToFileRanked:
		ap = evaluation.computeMyAPGoogle(value,objectDiscovered)
                if ap > maxAP:
                        maxAP = ap
	print(maxAP)
        sys.stdout.flush()
	return maxAP
	

# Folder paths
objectsRankingFile = 'allObjetsRankingFile.pickle'
groundTruthFile = 'googleGroundTruth.pickle'

model = smh.listdb_load(Config.MODEL_FILE)
print('Cargue modelo')
sys.stdout.flush()
ifs = smh.listdb_load(Config.INVERT_INDEX_FILE)
print('Cargue ifs')
sys.stdout.flush()
allObjetsToFileRanked = []
with open(objectsRankingFile, 'rb') as handle:
    allObjetsToFileRanked= pickle.load(handle)

print('Size of all objects discovered: {}'.format(len(allObjetsToFileRanked)))

groundTruthImages = {}
with open(groundTruthFile, 'rb') as handle:
    groundTruthImages= pickle.load(handle)

print('Size of allGroundTruthImages: {}'.format(len(groundTruthImages)))
sys.stdout.flush()
allObjetsAP = []
averageAP = 0.0

with poolcontext(processes=multiprocessing.cpu_count()-1) as pool:
        results = pool.map(partial(paralelFunc, allObjetsToFileRanked=allObjetsToFileRanked), groundTruthImages.values())

for result in results:
	averageAP+=result

print('Average AP: {}'.format(averageAP/14304.0))
