#!/usr/bin/env python2

import tensorflow as tf

frozen_graph = './model.ckpt.data-00000-of-00001'
with tf.gfile.GFile(frozen_graph, 'rb') as f:
	restored_graph_def = tf.GraphDef()
	restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")


