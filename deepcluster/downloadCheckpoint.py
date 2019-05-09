import tensorflow as tf
from datasets import dataset_utils

url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
checkpoints_dir = '/mnt/data/visual_instance_mining/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
