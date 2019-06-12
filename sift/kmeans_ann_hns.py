
import numpy as np
import os
import sys
import io
import random
import time
import tqdm
import hnswlib
import re

DIM=128
MAX_ITER = 30
CLUSTER_NUM = 150000
SIFT_FEATURES = 44039833

def count_sift_features(inputpath):
    print('Read sift Features')
    start_time = time.time()
    non_decimal = re.compile(r'[^\d]+')

    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    index = 0
    fileIndex = 0
    for entry in filelist:
        # Read features
        fileObj = open(inputpath+entry,'r')
        fileObj.readline()
        lineStrip = fileObj.readline().rstrip()
        m = int(lineStrip)
        for counter in range(m):
                data = fileObj.readline().rstrip()
                index+=1
        print(index)
        fileIndex+=1
        print(fileIndex)

    print('Read delf fetures Total time: %.3f s' % (time.time() - start_time))

def read_sift_features(inputpath):
    print('Read sift Features')
    start_time = time.time()
    non_decimal = re.compile(r'[^\d]+')

    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    allDesc = np.empty((SIFT_FEATURES,DIM))
    index = 0
    fileIndex = 0
    for entry in filelist:
	fileIndex+=1
        # Read features
	fileObj = open(inputpath+entry,'r')
        fileObj.readline()
  	lineStrip = fileObj.readline().rstrip()
	m = int(lineStrip)
	#print(entry)
	for counter in range(m):
		data = fileObj.readline().rstrip()
		dataSplit = data.split(' ')
		dataSplit = dataSplit[5:]
		desc = map(int,dataSplit)
        	allDesc[index:index+1,:] = desc
	        index+=1
	#print(index)
	#print(fileIndex)
	#print(entry)

    print('Read delf fetures Total time: %.3f s' % (time.time() - start_time))
    return allDesc

def get_random_clusters(sift_features):
    print('Get random clusters')
    idx = random.sample(range(SIFT_FEATURES), CLUSTER_NUM)
    return sift_features[idx,:]

def kMeans(delf_features,clusters):
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
		for feature in delf_features:
		    labels, distances = p.knn_query(feature, k=1)
		    new_centers[labels[0,0]] += feature
		    clus_size[labels[0,0]]+=1
                    index+=1
                    sys.stdout.write("\r Percent : %.3f" % (index/float(SIFT_FEATURES)))
                    sys.stdout.flush()
                print('\n')
                print('Re-assing cluster')
                for j in range(CLUSTER_NUM):
		    if clus_size[j] > 0:
			clusters[j] = new_centers[j] / clus_size[j]
		    else:
			rval = random.randint(0, SIFT_FEATURES-1)
			clusters[j] = delf_features[rval]
			print('Empty cluster replaced')
                if i==MAX_ITER-1:
                    p.save_index("hsm_150000_sift_30iter_20lan_500_balanc.bin")

	print('Total time: %.3f s' % (time.time() - start_time)) 

def main(inputpath):
    if os.path.isdir(inputpath):
        #count_sift_features(inputpath)
        delf_features = read_sift_features(inputpath)
	clusters = get_random_clusters(delf_features)
        kMeans(delf_features,clusters)

    else:
        print "File doesn't exist"
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "Uso: python kmeans_ann.py features/"
