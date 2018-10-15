import tensorflow as tf
import numpy as np
import sys
import os


class S2parser():
    """ defined the Sentinel 2 .tfrecord format """

    def __init__(self):
        self.feature_format = {
            'x10/data': tf.FixedLenFeature([], tf.string),
            'x10/shape': tf.FixedLenFeature([4], tf.int64),
            'x20/data': tf.FixedLenFeature([], tf.string),
            'x20/shape': tf.FixedLenFeature([4], tf.int64),
            'x60/data': tf.FixedLenFeature([], tf.string),
            'x60/shape': tf.FixedLenFeature([4], tf.int64),
            'dates/doy': tf.FixedLenFeature([], tf.string),
            'dates/year': tf.FixedLenFeature([], tf.string),
            'dates/shape': tf.FixedLenFeature([1], tf.int64),
            'labels/data': tf.FixedLenFeature([], tf.string),
            'labels/shape': tf.FixedLenFeature([2], tf.int64)
        }

        return None

    def write(self, filename, x10, x20, x60, doy, year, labels):
        # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

        writer = tf.python_io.TFRecordWriter(filename)

        # Changed from 64bit to smaller sizes
        x10 = x10.astype(np.uint16)
        x20 = x20.astype(np.uint16)
        x60 = x60.astype(np.uint16)
        doy = doy.astype(np.uint16)
        year = year.astype(np.uint16)
        labels = labels.astype(np.uint8)

        # Create a write feature
        feature = {
            'x10/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x10.tobytes()])),
            'x10/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x10.shape)),
            'x20/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x20.tobytes()])),
            'x20/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x20.shape)),
            'x60/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x60.tobytes()])),
            'x60/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x60.shape)),
            'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
            'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
            'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
            'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
            'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def parse_example(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.parse_single_example(serialized_example, self.feature_format)
        # decode and reshape x10
        x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.uint16), tf.cast(feature['x10/shape'], tf.int32))

        x20 = tf.reshape(tf.decode_raw(feature['x20/data'], tf.uint16), tf.cast(feature['x20/shape'], tf.int32))
        x60 = tf.reshape(tf.decode_raw(feature['x60/data'], tf.uint16), tf.cast(feature['x60/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.uint16), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.uint16), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.uint8), tf.cast(feature['labels/shape'], tf.int32))

        return x10, x20, x60, doy, year, labels


def test():
    print "Running self test:"
    print "temporary tfrecord file is written with random numbers"
    print "tfrecord file is read back"
    print "contents are compared"

    filename = "tmptile.tfrecord"

    # create dummy dataset
    x10 = (np.random.rand(5, 24, 24, 4) * 1e3).astype(np.uint16)
    x20 = (np.random.rand(5, 12, 12, 4) * 1e3).astype(np.uint16)
    x60 = (np.random.rand(5, 4, 4, 3) * 1e3).astype(np.uint16)
    labels = (np.random.rand(24, 24)).astype(np.uint8)
    doy = (np.random.rand(5) * 1e3).astype(np.uint16)
    year = (np.random.rand(5) * 1e3).astype(np.uint16)

    # init parser
    parser = S2parser()

    parser.write(filename, x10, x20, x60, doy, year, labels)

    tf_record_file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, s = reader.read(tf_record_file_queue)

    x10_, x20_, x60_, doy_, year_, labels_ = parser.parse_example(s)
    with tf.Session() as sess:
        tf.global_variables_initializer()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        x10_, x20_, x60_, doy_, year_, labels_ = sess.run([x10_, x20_, x60_, doy_, year_, labels_ ])

        if np.all(x10_ == x10) and np.all(x20_ == x20) and np.all(x60_ == x60) and np.all(labels_ == labels) and np.all(
                doy_ == doy) and np.all(year_ == year):
            print "PASSED"
        else:
            print "NOT PASSED"

    os.remove(filename)


if __name__ == '__main__':
    test()
