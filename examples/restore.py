#!/usr/bin/env python2

import tensorflow as tf

# Create some variables
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
	# Restore variables from disk
	# There is not a physical file called /tmp/model.ckpt. It is the prefix of filenames created for the checkpoint
	saver.restore(sess, "./model.ckpt")
	print("Model restored.")
	# check the values of the variables
#	print("v1: %s" % v1.eval())
	print(sess.run(v1))
#	print("v2: %s" % v2.eval())
	print(sess.run(v2))
