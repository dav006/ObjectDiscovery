
import numpy as np
import os
import sys
import io
import random
import time
import tqdm
import hnswlib

DIM=2048
MAX_ITER = 30
#DELF_FEATURES = 3810180
CONV_FEATURES = 980000

def read_features():
    start_time = time.time()
  
    allDesc = np.load('X_data_conv.npy')
    finalAllDesc = np.zeros(shape=(CONV_FEATURES,DIM))
    index=0
    for image in allDesc:
	for row in image:
		for desc in row:
			finalAllDesc[index]= desc
			index+=1
    print('Read fetures Total time: %.3f s' % (time.time() - start_time))
    return finalAllDesc
def get_random_clusters(features,cluster_num):
    print('Get random clusters')
    total_features = features.shape[0]
    idx = random.sample(range(total_features), cluster_num)
    return features[idx,:]

def kMeans(features,clusters,MY_DIM,cluster_num):
	start_time = time.time()
	total_features = features.shape[0]

	for i in tqdm.trange(MAX_ITER):
                print('Build Tree')
		p = hnswlib.Index(space='l2', dim=MY_DIM)
                p.init_index(max_elements=cluster_num, ef_construction=100, M=16)
                p.add_items(clusters)
		clus_size = np.zeros(cluster_num)
		new_centers = np.zeros((cluster_num,MY_DIM))
                
                print('Search KNN')
                index = 0
		for feature in features:
		    labels, distances = p.knn_query(feature, k=1)
		    new_centers[labels[0,0]] += feature
		    clus_size[labels[0,0]]+=1
                    index+=1
		    if index%1000 is 0:
                    	sys.stdout.write("\r Percent : %.3f" % (index/float(total_features)))
                    	sys.stdout.flush()
                print('\n')
                print('Re-assing cluster')
		emptyClus = 0
                for j in range(cluster_num):
		    if clus_size[j] > 0:
			clusters[j] = new_centers[j] / clus_size[j]
		    else:
			rval = random.randint(0, total_features-1)
			clusters[j] = features[rval]
			emptyClus+=1
			#print('Empty cluster replaced')
                print 'Empty clusters : '+str(emptyClus)
                if i==MAX_ITER-1:
                    p.save_index("hsm_25000_30iter_google_40_balan.bin")

	print('Total time: %.3f s' % (time.time() - start_time))
    
def main():
	features = read_features()
	clusters = get_random_clusters(features)
        kMeans(features,clusters,DIM)

def generateVocab(features,PCA_DIM,cluster_num):
	clusters = get_random_clusters(features,cluster_num)
	kMeans(features,clusters,PCA_DIM,cluster_num)
      
if __name__ == "__main__":
        main()
