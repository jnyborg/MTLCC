import numpy as np
from S2parser import S2parser
import os
import gzip
import shutil
import tensorflow as tf

parser=S2parser()
print("HI")

directory = "data_IJGI18/datasets/demo/480/data16"
filepaths=["{}/{}.tfrecord.gz".format(directory, 95)]
print(filepaths)


dataset = tf.data.TFRecordDataset(filepaths, compression_type="GZIP")

def normalize(serialized_feature):
    """ normalize stored integer values to floats approx. [0,1] """
    x10, x20, x60, doy, year, labels = serialized_feature
    x10 = tf.scalar_mul(1e-4, tf.cast(x10, tf.float32))
    x20 = tf.scalar_mul(1e-4, tf.cast(x20, tf.float32))
    x60 = tf.scalar_mul(1e-4, tf.cast(x60, tf.float32))
    doy = tf.cast(doy, tf.float32) / 365
    year = tf.cast(year, tf.float32) - 2016

    return x10, x20, x60, doy, year, labels

def mapping_function(serialized_feature):
    # read data from .tfrecords
    serialized_feature = parser.parse_example(serialized_feature)
    return normalize(serialized_feature)

print("applying the mapping function on all samples (will read tfrecord file and normalize the values)")
dataset = dataset.map(mapping_function)

print("repeat forever until externally stopped")
dataset = dataset.repeat()

print("combine samples to batches")
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))

print("make iterator")
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    print("retrieving one sample as numpy array (just for fun)")
    x10, x20, x60, doy, year, labels = sess.run(iterator.get_next())
    print(doy)
    print(x10.shape)
    print(x20.shape)
    print(x60.shape)
    print(doy.shape)
    print(year.shape)
    print(labels.shape)




