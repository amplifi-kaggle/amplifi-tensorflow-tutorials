"""When to use FixedLenFeature or VarLenFeature?

You can load images probably because you saved them using feature type tf.train.BytesList()
and whole image data is one big byte value inside a list.

If I'm right you're using tf.decode_raw to get the data out of the image you load from TFRecord.

Regarding example use cases: I use VarLenFeature for saving datasets for object detection task:
There's variable amount of bounding boxes per image (equal to object in image) therefore I need
another feature objects_number to track abount of objects (and bboxes). Each bounding box itself
is a list of 4 float coordinates.
"""

features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                # Label part
                'objects_number': tf.FixedLenFeature([], tf.int64), 
                'bboxes': tf.VarLenFeature(tf.float32),
                'labels': tf.VarLenFeature(tf.int64),
                # Dense data
                'image_raw': tf.FixedLenFeature([], tf.string)
            })

# Get metadata
objects_number = tf.cast(features['objects_number'], tf.int32)
height = tf.cast(features['height'], tf.int32)
width = tf.cast(features['width'], tf.int32)
depth = tf.cast(features['depth'], tf.int32)

# Actual data
image_shape = tf.parallel_stack([height, width, depth])
bboxes_shape = tf.parallel_stack([objects_number, 4])

# BBOX data is actually dense convert it to dense tensor
bboxes = tf.sparse_tensor_to_dense(features['bboxes'], default_value=0)
# Since information about shape is lost reshape it
bboxes = tf.reshape(bboxes, bboxes_shape)

image = tf.decode_raw(features['image_raw'], tf.int8)
image = tf.reshape(image, image_reshape)

'''
Notice that "image_raw" is fixed length Feature (has one element) and holds values of types "bytes",
however a value of "bytes" type can itself have variable size (its a string of bytes, and can have
many symbols within it). So "image_raw" is a list with ONE element of type "bytes", which can 
be super big.

To further elaborate on how it works: Features are lists of values, those values have specific "type".

Datatypes for features are subset of data types for tensors, you have:
 - int64 (64 bit space in memory)
 - bytes (occupies as many bytes in memory as you want)
 - float (occupies 32-64 bits in memory idk how much)

So you can store variable length data without VarLenFeatures at all (actually well you do it),
but first you would need to convert it into bytes/string feature, and then decode it.

And this is most common method.
'''
