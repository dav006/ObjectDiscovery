
import numpy as np
import os
import sys
import io
import random
import time
import tqdm
import hnswlib
sys.path.append('../common/')
import dumpy

DIM=128
MAX_ITER = 30
CLUSTER_NUM = 1000000
LIFT_FEATURES = 4722662

def count_lift_features(inputpath):
    print('Read lift Features')
    start_time = time.time()

    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    index = 0
    count = 0
    for entry in filelist:
        # Read features
        fileObj = dumpy.loadh5(inputpath+entry)
        fileObj = fileObj['descriptors']
        size = len(fileObj)
        index+=size
	count+=1
        print(index)
        print(count)

    print('Read lift fetures Total time: %.3f s' % (time.time() - start_time))
    print(index)

def read_lift_features(inputpath):
    print('Read lift Features')
    start_time = time.time()

    filelist = sorted(os.listdir(inputpath+'.'))
    random.shuffle(filelist)
    allDesc = np.empty((LIFT_FEATURES,DIM))
    index = 0
    for entry in filelist:
        # Read features
	fileObj = dumpy.loadh5(inputpath+entry)
        fileObj = fileObj['descriptors']
	size = len(fileObj)
	allDesc[index:index+size,:] = fileObj
        index+=size

    print('Read lift fetures Total time: %.3f s' % (time.time() - start_time))
    return allDesc

def get_random_clusters(sift_features):
    print('Get random clusters')
    idx = random.sample(range(LIFT_FEATURES), CLUSTER_NUM)
    return sift_features[idx,:]

def kMeans(lift_features,clusters):
	start_time = time.time()

	for i in tqdm.trange(MAX_ITER):
                print('Build Tree')
		p = hnswlib.Index(space='l2', dim=DIM)
                p.init_index(max_elements=CLUSTER_NUM, ef_construction=250, M=16)
                p.add_items(clusters)
		clus_size = np.zeros(CLUSTER_NUM)
		new_centers = np.zeros((CLUSTER_NUM,DIM))
                
                print('Search KNN')
                index = 0
		for feature in lift_features:
		    labels, distances = p.knn_query(feature, k=1)
		    new_centers[labels[0,0]] += feature
		    clus_size[labels[0,0]]+=1
                    index+=1
                    sys.stdout.write("\r Percent : %.3f" % (index/float(LIFT_FEATURES)))
                    sys.stdout.flush()
                print('\n')
                print('Re-assing cluster')
                for j in range(CLUSTER_NUM):
		    if clus_size[j] > 0:
			clusters[j] = new_centers[j] / clus_size[j]
		    else:
			rval = random.randint(0, LIFT_FEATURES-1)
			clusters[j] = lift_features[rval]
			print('Empty cluster replaced')
                if i==MAX_ITER-1:
                    p.save_index("hsm_1000000_lift_30iter.bin")

	print('Total time: %.3f s' % (time.time() - start_time)) 

def main(inputpath):
    if os.path.isdir(inputpath):
        #count_lift_features(inputpath)
        lift_features = read_lift_features(inputpath)
	clusters = get_random_clusters(lift_features)
        kMeans(lift_features,clusters)

    else:
        print "File doesn't exist"
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "Uso: python kmeans_ann.py features/"
