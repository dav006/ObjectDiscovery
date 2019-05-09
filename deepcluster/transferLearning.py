import sys
import os

import tensorflow as tf

from datasets import imagenet
from nets import resnet_v1
from preprocessing import vgg_preprocessing

import glob

# constants
DATASET_DIR = "../../flower_photos/"
_FILE_PATTERN = 'flowers_%s_*.tfrecord'

batch_size = 16
num_epochs = 1

is_training = True 
checkpoints_dir = '/mnt/data/visual_instance_mining/checkpoints'

slim = tf.contrib.slim

image_size = resnet_v1.resnet_v1.default_image_size

#Returns number of lines in text file
#used to determine number of classes in labels.txt file
def file_len(filename):
    return sum(1 for line in open(filename))

# Return number of records in list of filenames
def get_num_records_tfecords(filenames):
    c = 0
    for fn in filenames:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c



def _parse_function(example_proto):
    features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature((), tf.int64),
      'image/width': tf.FixedLenFeature((), tf.int64),
    }
    
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"])
    width = tf.cast(parsed_features["image/width"], tf.int32)
    height = tf.cast(parsed_features["image/height"], tf.int32)
    label = tf.cast(parsed_features["image/class/label"], tf.int32)

    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, 3])
    
    #Images need to have the same dimensions for feeding the network
    image = vgg_preprocessing.preprocess_image(image, image_size, image_size)

    return image, label


# Directory to save summaries to
logdir = "logs/"
if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)

graph = tf.Graph()
with graph.as_default():
	tf.logging.set_verbosity(tf.logging.INFO)

	# Load datasets
	print("Loading dataset")
    	train_filenames = glob.glob(DATASET_DIR+_FILE_PATTERN % ("train"))
    	train_dataset = tf.data.TFRecordDataset(train_filenames)
    	train_dataset = train_dataset.map(_parse_function)
    	train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    	batched_train_dataset = train_dataset.batch(batch_size)

    	val_filenames = glob.glob(DATASET_DIR+_FILE_PATTERN % ("validation"))
    	val_dataset = tf.data.TFRecordDataset(val_filenames)
    	val_dataset = val_dataset.map(_parse_function)
    	batched_val_dataset = val_dataset.batch(batch_size)

    	num_classes = file_len(os.path.join(DATASET_DIR,"labels.txt"))
    	num_train_records = get_num_records_tfecords(train_filenames)
    	print("Loaded train dataset with %d images belonging to %d classes" % (num_train_records, num_classes))
    	num_batches = np.ceil(num_train_records/batch_size)
    
    	num_val_records = get_num_records_tfecords(val_filenames)
    	print("Loaded val dataset with %d images belonging to %d classes" % (num_val_records, num_classes))
    
    	#iterator
    	iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                        batched_train_dataset.output_shapes)
    	images, labels = iterator.get_next()

    	train_init_op = iterator.make_initializer(batched_train_dataset)
    	val_init_op = iterator.make_initializer(batched_val_dataset)	

	with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.00001)):
		logits, _ = resnet_v1.resnet_v1_50(images,num_classes,is_training=is_training)
	
	logits = tf.squeeze(logits)

	exclude = ["resnet_v1_50/logits", "resnet_v1_50/AuxLogits"]
	variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

	# Restore the remaining variables
	init_fn = slim.assign_from_checkpoint_fn(
		os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
	        variables_to_restore)

	logits_variables = slim.get_variables("resnet_v1_50/logits") + slim.get_variables("resnet_v1_50/AuxLogits")
	logits_init = tf.variables_initializer(logits_variables)
	
	# Loss function:
	predictions = tf.to_int32(tf.argmax(logits, 1))
    	tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    	total_loss = tf.losses.get_total_loss()

	temp = set(tf.all_variables())
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
	logits_train_op = optimizer.minimize(total_loss, var_list=logits_variables) #use this op to only train the last layer
	full_train_op = optimizer.minimize(total_loss)                              #use this op to train the whole network

    #this needs to come after defining the training op
    	adam_init_op = tf.initialize_variables(set(tf.all_variables()) - temp)
    
    # Define the metric and update operations (taken from http://ronny.rest/blog/post_2017_09_11_tf_metrics/)
    	tf_metric, tf_metric_update = tf.metrics.accuracy(labels, predictions, name="accuracy_metric")

    	# Isolate the variables stored behind the scenes by the metric operation
    	running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")

    	# Define initializer to initialize/reset running variables
    	running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    	acc_summary = tf.summary.scalar('accuracy', tf_metric)

    	# To save the trained model
    	saver = tf.train.Saver()

    	tf.get_default_graph().finalize()

with tf.Session(graph=graph) as sess:
    #Initializations
    init_fn(sess)
    sess.run(logits_init)
    sess.run(adam_init_op)
    
    print("Writing summaries to %s" % logdir)
    train_writer = tf.summary.FileWriter(os.path.join(logdir,"train/"), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(logdir,"valid/"), sess.graph)
 
    #Training
    for epoch in range(num_epochs):
        print('Starting training epoch %d / %d' % (epoch + 1, num_epochs))
        # initialize the iterator with the training set
        sess.run(train_init_op)

        pbar = tqdm(total=num_batches)  #progress bar showing how many batches remain
        while True:
            try:
                # train on one batch of data
                _ = sess.run(full_train_op)
                pbar.update(1)

            except tf.errors.OutOfRangeError:
                break
        pbar.close()

        # Compute training and validation accuracy
        sess.run(train_init_op)
        # initialize/reset the accuracy running variables
        sess.run(running_vars_initializer)

        while True:
            try:
                sess.run(tf_metric_update)
            except tf.errors.OutOfRangeError:
                break
        train_acc = sess.run(tf_metric)
        summary = sess.run(acc_summary)
        print('Train accuracy: %f' % train_acc)
        train_writer.add_summary(summary,epoch +1)
        train_writer.flush()

        sess.run(val_init_op)

        # initialize/reset the accuracy running variables
        sess.run(running_vars_initializer)
        
        while True:
            try:
                sess.run(tf_metric_update)
            except tf.errors.OutOfRangeError:
                break
        # Calculate the score
        val_acc = sess.run(tf_metric)
        summary = sess.run(acc_summary)
        print('Val accuracy: %f' % val_acc)
        val_writer.add_summary(summary,epoch +1)
        val_writer.flush()

    #Save model
    saver.save(sess, os.path.join(logdir, "trained_model.ckpt" ))


