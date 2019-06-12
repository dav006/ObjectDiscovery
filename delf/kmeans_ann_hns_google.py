
import numpy as np
import os
import sys
import io
import random
from delf import feature_io
import time
import tqdm
import hnswlib

DIM=40
MAX_ITER = 30
CLUSTER_NUM = 150000
#DELF_FEATURES = 387367746
DELF_FEATURES = 183546693
SIZE_SPLIT = 2
SIZE_FILES_SPLIT = 104232
data_split = [(0,104232),(104232,208463)]

def count_delf(inputpath):
    print('Read delf Features')
    start_time = time.time()
    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    index = 0
    count = 0
    for entry in filelist:
        # Read features
        _, _, descriptors, _, _ = feature_io.ReadFromFile(inputpath+entry)
        size =descriptors.shape[0]
        index+=size
	count+=1
        print(index)
	print(count)

    print('Read delf fetures Total time: %.3f s' % (time.time() - start_time))
    print(index)

def read_delf_features(inputpath):
    print('Read delf Features')
    start_time = time.time()

    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    
    idToFile = {}
    index = 0
    for entry in filelist:
	idToFile[index] = inputpath + entry
        index+=1	

    print('Read delf fetures Total time: %.3f s' % (time.time() - start_time))
    return idToFile

def randomFeature(rFile,idToFile):
    myFile = idToFile[rFile]
    _, _, descriptors, _, _ = feature_io.ReadFromFile(myFile)
    size =descriptors.shape[0]
    rFeature = random.randint(0, size-1)
    return descriptors[rFeature,:]

def get_random_clusters(idToFile):
    print('Get random clusters')
    global data_split
    features = np.zeros((CLUSTER_NUM,DIM))
    for i in range(CLUSTER_NUM):
	rSplit = random.randint(0, SIZE_SPLIT-1)
        start,end = data_split[rSplit]
        rFile = random.randint(start,end-1)
        feature = randomFeature(rFile,idToFile)
        features[i,:]=feature
        sys.stdout.write("\r Percent Cluster: %.3f" % (i/float(CLUSTER_NUM)))
        sys.stdout.flush()
    print('\n')
    return features

def loadDelfSplit(data_split_index,idToFile):
    global data_split
    start,end = data_split[data_split_index]
    features=np.empty((SIZE_FILES_SPLIT*1000,DIM))
    index=0
    print('\n')
    print('Load delf split : {}'.format(data_split_index))
    for i in range(start,end):
        myFile = idToFile[i]
        _, _, descriptors, _, _ = feature_io.ReadFromFile(myFile)
        size =descriptors.shape[0] 
        features[index:index+size,:] = descriptors
        index+=size
    return features[0:index,:]

def kMeans(idToFile,clusters):
        global data_split
	start_time = time.time()

	for i in tqdm.trange(MAX_ITER):

                print('Build Tree')
		p = hnswlib.Index(space='l2', dim=DIM)
                p.init_index(max_elements=CLUSTER_NUM, ef_construction=100, M=16)
                p.add_items(clusters)
		clus_size = np.zeros(CLUSTER_NUM)
		new_centers = np.zeros((CLUSTER_NUM,DIM))
                
                print('Search KNN')
                index = 0
		for data_split_index in range(SIZE_SPLIT):
                    features = loadDelfSplit(data_split_index,idToFile)
                    for feature in features:
		        labels, distances = p.knn_query(feature, k=1)
		        new_centers[labels[0,0]] += feature
		        clus_size[labels[0,0]]+=1
                        index+=1
                        if index%1000 is 0:
                            sys.stdout.write("\r Percent : %.3f" % (index/float(DELF_FEATURES)))
                            sys.stdout.flush()
                    del features
                
                print('\n')
                print('Re-assing cluster')
                for j in range(CLUSTER_NUM):
		    if clus_size[j] > 0:
			clusters[j] = new_centers[j] / clus_size[j]
		    else:
			rSplit = random.randint(0, SIZE_SPLIT-1)
                        rFile = random.randint(data_split[rSplit][0],data_split[rSplit][1]-1)
			feature = randomFeature(rFile,idToFile)
			clusters[j] = feature
			print('Empty cluster replaced')
                if i==MAX_ITER-1:
                    p.save_index("hsm_150000_30iter_google_11_desba.bin")

	print('Total time: %.3f s' % (time.time() - start_time))
    
def main(inputpath):
    if os.path.isdir(inputpath):
	'''	
        idToFile = read_delf_features(inputpath)
        clusters = get_random_clusters(idToFile)
        kMeans(idToFile,clusters)
	'''
	count_delf(inputpath)

    else:
        print "File doesn't exist"
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "Uso: python kmeans_ann.py features/"
