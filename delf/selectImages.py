import smh
from collections import Counter
import pickle
import operator
from config import Config
from delf import feature_io
import hnswlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3


# Create an S3 client
s3 = boto3.client('s3')

model = smh.listdb_load(Config.MODEL_FILE)
ifs = smh.listdb_load(Config.INVERT_INDEX_FILE)
idToFileName = ()
with open('indexToFile.pickle', 'rb') as handle:
    idToFileName= pickle.load(handle)

groundTruthFile = 'googleGroundTruth_selected.pickle'
groundTruthImages = {}
with open(groundTruthFile, 'rb') as handle:
    groundTruthImages= pickle.load(handle)


# Create array with all images associated with a objectDiscovered
objectDiscovered = model.ldb[74]
imageCount={}
for visualWord in objectDiscovered:
	visualWordName = visualWord.item
        for image in ifs.ldb[visualWordName]:
		if image.item not in imageCount:
			imageCount[image.item] = 0
                imageCount[image.item] += 1

sort = sorted(imageCount.items(), key=lambda x: x[1], reverse=True)
'''
for i in range(len(sort)):
	if idToFileName[sort[i][0]] not in groundTruthImages['10900']:
		print(i)
		break

567
'''	
idLand = 0
print(sort[idLand])
fileName = idToFileName[sort[idLand][0]]
print(fileName in groundTruthImages['13526'])
print(fileName)

inputpath  = '../../google_landmark_attention_selected/'
image_path = '../../google_landmark_selected/'
locations, _, descriptors, _, _ = feature_io.ReadFromFile(inputpath+fileName+'.delf')
# Reiniting, loading the index
p = hnswlib.Index(space='l2', dim=40)
p.load_index("150000/hsm_150000clu_30iter_google_20land_2000balan.bin", max_elements = 150000)
labels, _ = p.knn_query(descriptors, k=1)

indexArray = []
for i in range(descriptors.shape[0]):
	for visualWord in objectDiscovered:
        	visualWordName = visualWord.item
		if labels[i] == visualWordName:
			indexArray.append(i)
                        break

print(len(indexArray))
fig = plt.figure(frameon=False)
img_1 = mpimg.imread(image_path+fileName+'.jpg')

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

im = ax.imshow(img_1, cmap='gray',aspect='normal')
ax.scatter(locations[indexArray, 1],locations[indexArray, 0], color='red',alpha=0.5)

plt.savefig(fileName+'.jpg')
# files automatically and upload parts in parallel.
s3.upload_file(fileName+'.jpg', 'davtempbuck', fileName+'.jpg')
