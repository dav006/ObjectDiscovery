# Uses SMH to create inverted index
from config import Config
import sys
sys.path.append('../common/')
import smhHelper 

smhHelper.createInvertedIndex(Config.CORPUS_FILE,Config.INVERT_INDEX_FILE)
