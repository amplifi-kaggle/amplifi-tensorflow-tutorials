import tensorflow as tf

"""Obtaining total number of records from .tfrecords file in Tensorflow?

No it is not possible. TFRecord does not store any metadata about the data being stored inside.
This file:
 represents a sequence of (binary) strings. The format is not random access, so it is suitable for streaming 
 large amounts of data but not suitable if fast sharding or other non-sequential access is desired.

If you want, you can store this metadata manually or use a record_iterator to get the number (you will need
to iterate through all the records that you have:)

"""
sum(1 for _ in tf.python_io.tf_record_iterator(file_name))

"""To count the number of records, you should be able to use tf.python_io.tf_record_iterator.

        To just keep track of the model training, tensorboard comes in handy.
"""
c = 0
for fn in tf_records_filenames:
    for record in tf.python_io.tf_record_iterator(fn):
        c += 1

# TFrecords Guide : http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""Binary files are sometimes easier to use, because you don't have to specify different
directories for images and groundtruth annotations. While storing your data in binary file,
you have your data in one block of memory, compared to storing each image and annotation separately.

Opening a file is a considerably time-consuming operation especially if you use hdd and not ssd,
because it involves moving the disk reader head and that takes quite some time. Overall, by using
binary files you make it easier to distribute and make the data better aligned for efficient reading.

The post consists of three parts: in the first part, we demonstrate how you can get raw data bytes
of any image using numpy which is in some sense similar to what you do when converting your dataset
to binary format. Second part shows how to convert a dataset to tfrecord file without defining
a computational graph and only by employing some built-in tensorflow functions. Third part explains
how to define a model for reading your data from created binary file and batch it in a random manner,
which is necessary during training.

The blog post is created using jupyter notebook. After each chunk of a code you can see the result
of its evaluation. You can also get the notebook file from here.



"""
# Getting raw data bytes in numpy
import numy as np
import skimage.io as io

cat_img = io.imread('cat.jpg')
io.imshow(cat_img)

# Let's convert the picture into string representation
# using the ndarray.tostring() function
cat_string = cat_img.tostring()

# Now let's convert the string back to the image
# Important: the dtype should be specified
# otherwise the reconstruction will be errorness
# Reconstruction is 1d, so we need sizes of image to fully reconstruct it
reconstructed_cat_1d = np.fromstring(cat_string, dtype=np.uint8)

# Here we reshape the 1d representation
# This is the why we need to store the sizes of image
# along with its serialized representation.
reconstructed_cat_img = reconstructed_cat_1d.reshape(cat_img.shape)

# Let's check if we got everything right and compare
# reconstructed array to the original one.
np.allclose(cat_img, reconstructed_cat_img)


# Get some image/annotation pairs for example 
filename_pairs = [
('/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg',
'/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'),
('/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg',
'/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png'),
('/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg',
'/home/dpakhom1/tf_projects/segmentation/VOCdevkit/VOCdevkit/VOC2012/SegmentationClass/2007_000063.png')
]

"""Important: We are using PIL to read .png file later.
This was done on purpose to read indexed png files in a special way
-- only indexes and not map the indexes to actual rgb values. This is specific 
to PASCAL VOC dataset data. If you don't want this type of behavior 
consider using skimage.io.imread()
"""
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare to the reconstructed ones
original_images = []

for img_path, annotation_path in filename_pairs:
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))
    
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes of images
    # to later read raw serialized string, convert to 1d array and convert to
    # respective shape that image used to have.
    
    height = img.shape[0]
    width = img.shape[1]
    
    # Put in the original images into array
    # Just for future check for correctness
    original_images.append((img, annotation))
    
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    
    feature = {
         'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    writer.write(example.SerializeToString())
    
writer.close()

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                    .in64_list
                    .value[0])
    width = int(example.features.feature['width']
                    .in64_list
                    .value[0])
    img_string = (example.features.feature['image_raw']
                    .bytes_list
                    .value[0])
    
    annotation_string = (example.features.feature['mask_raw']
                    .bytes_list
                    .value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    
    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))
    reconstructed_images.append((reconstructed_img, reconstructed_annotation))
    
# Let's check if the reconstructed images match the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))


import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # (mnist.IMAGE_PIXELS).
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.pack([height, width, 3])
    annotation_shape = tf.pack([height, width, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                          target_height=IMAGE_HEIGHT,
                                                          target_width=IMAGE_WIDTH)
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                                               target_height=IMAGE_HEIGHT,
                                                               target_width=IMAGE_WIDTH)
    
    images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                                batch_size=2,
                                                                capacity=30,
                                                                num_threads=2,
                                                                min_after_dequeue=10)
    
    return images, annotations

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

# Even when reading in multiple threads, share the filename queue.
image, annotation = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Let's read off 3 batches just for example
    for i in xrange(3):
        img, anno = sess.run([image, annotation])
        print(img[0, :, :, :].shape)
        
        print('current batch')
        
        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random
        io.imshow(img[0, :, :, :])
        io.show()
        
        # 1st segmentation label from batches 
        io.imshow(anno[0, :, :, :])
        io.show()
        
        io.imshow(img[1, :, :, :])
        io.show()
        
        # 2nd segmentation label from batches 
        io.imshow(anno[1, :, :, 0])
        io.show()
        
    coord.request_stop()
    coord.join(threads)

"""

Q. Why write the image in bytes/string format, and not int64 or something else?
And why do you write height/width to record in int64 then recast to int32 when decoding the record?

A. int64_list allows you to serialize integers or a list of integers. However, it will fail if you try to serialize something
more complex - e.g. a list of lists (image). Therefore, converting your images into a string is a better choice.
The only problem with this approach is restoration process. Effectively, once you convert your
string back into integers, you get a list of integers without any shape. However, you can restore
the shape, if you know dimensions! Thus, we need to restore the dimensions of the image.
tf.reshape takes images dimensions only in tf.in32 format.

tf.pack no longer exists in Tensorflow 1.0. Instead, it should be changed to tf.stack.

"""

