# SMH WITH SIFT descriptors
Unsupervised object discovery from large-scale image collections using sift descriptors

Downloads:
1. Visual word ids and geometry: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ to folder : data/word_oxc1_hesaff_sift_16M_1M
2. Ground truth files:http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ to folder: gt_files_170407/

Pre-requisits:
1. Follow instructions to build and install SMH https://github.com/gibranfp/Sampled-MinHashing

Instructions:
1. run SMH/idToFile.py to generate mappings from fileId to fileName
2. run SMH/toBagOfWords.py to generate binary bag words without stop words
3. run SMH/toInvertIndex.py to generate inverted index
4. run SMH/createModel.py to create object model
5. run SMH/rankingImages.py to rank images for evaluation
5. run SMH/evaluate.py to get AP for each landmark
