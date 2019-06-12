from config import Config
import sys
sys.path.append('../common/')
import smhHelper 

smhHelper.createModel(Config.CORPUS_FILE,Config.INVERT_INDEX_FILE,Config.MODEL_FILE,int(sys.argv[1]))
