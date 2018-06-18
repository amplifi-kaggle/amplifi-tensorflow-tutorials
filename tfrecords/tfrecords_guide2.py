"""
In my case, if the network is small, then the execution in GPU is faster
that the data loading, which leads to idle GPU time. So I want to the queue for 
fetching data is always full.

https://stackoverflow.com/questions/45427637/numpy-to-tfrecords-is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrec/45428167#45428167
"""


# 1. Creation of tfrecords from a numpy array

def npy_to_tfrecords(...):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    # Loop through all the features you want to write
    for ...:
        let say X is of np.array([[...][...]])
        let say y is of np.array[[0/1]]

        # Feature contains a map of string to feature proto objects
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y))

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()

        # write the serialized object to the disk
        wrtier.write(serialized)

    writer.close()


# 2. Read the tfrecords using the Dataset API (tensorflow >= 1.2)

# Creates a dataset that reads all of the examples from filenames.
filenames = ["file1.tfrecord", "file2.tfrecord", ..."fileN.tfrecord"]

# Using Dataset API
dataset = tf.contrib.data.TFRecordDataset(filenames)

# example proto decode
def _parse_function(example_proto):
    keys_to_features = {'X': tf.FixedLenFeature((shape_of_npy_array), tf.float32),
                        'y': tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    return parsed_features['X'], parsed_featuers['y']

# Parse the record into tensors.
dataset = dataset.map(_parse_function)

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=10000)

# Repeat the input indefinitely
dataset = dataset.repeat()

# Generate batches
dataset = dataset.batch(batch_size)

# Create a one-shot iterator
iterator = dataset.make_one_hot_iterator()

# Get batch X and y
X, y = iterator.get_next()

