#!/usr/bin/env python3

"""Contains functions for evaluation and summarization of metrics.

The evaluation.py module contains helper functions for evaluating Tensorflow
modules using a variety of metrics and summarizing the results

Evaluating Metrics

In the simplest use case, we use a model to create the predictions, then specify
the metrics and finally call the 'evaluation' method:

# Create model and obtain the predictions:
images, labels = LoadData(...)
predictions = MyModel(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
		"accuracy": slim.metrics.accuracy(predictions, labels),
		"mse": slim.metrics.mean_squared_error(predictions, labels),
})


initial_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
	metric_values = slim.evaluation(
				sess,
				num_evals=1,
				initial_op=initial_op,
				eval_op=names_to_updates.values(),
				final_op=names_to_updates.values())

for metric, value in zip(names_to_values.keys(), metric_values):
	logging.info('Metric % has value: %f', metric, value)



"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow 
