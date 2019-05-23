# Uses SMH to create inverted index and model

import smh
from smh_prune import SMHD

def createInvertedIndex(CORPUS_FILE, INVERT_INDEX_FILE):
	print('open')
	corpus = smh.listdb_load(CORPUS_FILE)
	print('invert')
	ifs = corpus.invert()
	ifs.save(INVERT_INDEX_FILE)

def createModel(CORPUS_FILE,INVERT_INDEX_FILE,MODEL_FILE):
	corpus = smh.listdb_load(CORPUS_FILE)
	ifs = smh.listdb_load(INVERT_INDEX_FILE)

	discoverer = SMHD( 
		tuple_size = 2, 
		number_of_tuples = 433, 
		min_set_size = 3, 
		overlap = 0.8,
		min_cluster_size = 3,
		cluster_tuple_size = 3,
		cluster_number_of_tuples =  255,
		cluster_table_size=2**24
		)

	print('Iniciar fit')
	models = discoverer.fit(ifs, prune = False, expand = corpus)
	print('Terminar fit')
	models.save(MODEL_FILE)
