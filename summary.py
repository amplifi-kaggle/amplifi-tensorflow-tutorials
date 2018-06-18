#!/usr/bin/env python3

import tensorflow as tf

w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
w2 = tf.Varialbe(tf.truncated_normal(shape=[20]), name='w2')
tf.add_to_collection('vars', w1)
tf.add_to_collection('vars', w2)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my-model')

# Restore the model
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
	v_ = sess.run(v)
	print(v_)

