#!/usr/bin/env python2

import tensorflow as tf

# Create some variables
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

# Add ops to save and restore only "v2" using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that
with tf.Session() as sess:
	# Intialize v1 since the saver will not
	# If you only restore a subset of the model variables at the start of a session,
	# you have to run an initialize op for the other variables.
	v1.initializer.run()
	
	# Restore variables from disk
	saver.restore(sess, "./model.ckpt")
	print("Model restored.")
	# check the values of the variables
	print("v1: %s" % v1.eval())
	print("v2: %s" % v2.eval())

'''

By default, Saver uses the value of the tf.Variable.name property for each variable. However,
when you create a Saver object, you may optionally choose names for the variables in the checkpoint files.

'''
