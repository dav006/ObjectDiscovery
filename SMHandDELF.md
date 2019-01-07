# SMH WITH DELF descriptors
Unsupervised object discovery from large-scale image collections using DELF descriptors

Pre-requisits:
1. Follow instructions to build and install SMH https://github.com/gibranfp/Sampled-MinHashing
2. Follow instructions to extract DELF descriptors https://github.com/tensorflow/models/tree/master/research/delf

Instructions:
1. run delf/minibatchkmeans.py to generate visual vocabulary
2. run delf/idToFile.py to generate mappings from fileId to fileName
3. run delf/toBagOfWords.py to generate binary bag words without stop words
4. run delf/toInvertIndex.py to generate inverted index
5. run delf/createModel.py to create object model
6. run delf/rankingImages.py to rank images for evaluation
7. run delf/evaluate.py to get AP for each landmark
