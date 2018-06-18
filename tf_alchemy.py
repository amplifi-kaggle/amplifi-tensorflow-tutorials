#!/usr/bin/env python3

import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

sim = tf.contrib.slim


def configure_learning_rate(flags, num_samples_per_epoch, global_step):
	"""Configures the learning rate.

	Args:
		num_samples_per_epoch: The number of samples in each epoch of training.
		global_step: The global_step tensor.
	Returns:
		A 'Tensor' representing the learning rate.
	"""

	decay_steps = int(num_samples_per_epoch / flags.batch_size * flags.num_epochs_per_decay)

	if flags.learning_rate_decay_type == 'exponential':
		return tf.train.exponential_decay(flags.learning_rate,
											global_step,
											decay_steps,
											flags.learning_rate_decay_factor,
											staircase=True,
											name='exponential_decay_learning_rate')
	elif flags.learning_rate_decay_type == 'fixed':
		return tf.constant(flags.learning_rate, name='fixed_learning_rate')
	elif flags.learning_rate_decay_type == 'polynomial':
		return tf.train.polynomial_decay(flags.learning_rate,
										global_step,
										decay_steps,
										flags.end_learning_rate,
										power=1.0,
										cycle=False,
										name='polynomial_decay_learning_rate')
	else
		raise ValueError('learning_rate_decay_type [%s] was not recognized',
						flags.learning_rate_decay_type)


def configure_optimizer(flags, learning_rate):
	"""
	Args:
		learning_rate: A scalar or 'Tensor' learning rate.
	Returns:
		An instance of an optimizer.
	"""
	if flags.optimizer == 'adadelta':
		optimizer = tf.train.AdadeltaOptimizer(
					learning_rate,
					rho=flags.adadelta_rho,
					epsilon=flags.opt_epsilon)
	elif flags.optimizer == 'adagrad':
		optimizer = tf.train.AdagradOptimizer(
					learning_rate,
					initial_accumulator_value=flags.adagrad_initial_accumulator_value)
	elif flags.optimizer == 'ftrl':
		optimizer = tf.train.FtrlOptimizer(
					learning_rate,
					learning_rate_power=flags.ftrl_learning_rate_power,
					initial_accumulator_value=flags.ftrl_initial_accumulator_value,
					l1_regularization_strength=flags.ftrl_l1,
					l2_regularization_strength=flags.ftrl_l2)
	elif flags.optimizer == 'momentum':
		optimizer = tf.train.MomentumOptimizer(
					learning_rate,
					momentum=flags.momentum,
					name='Momentum')
	elif flags.optimizer == 'rmsprop':
		optimizer = tf.train.RMSPropOptimizer(
					learning_rate,
					decay=flags.rmsprop_decay,
					momentum=flags.rmsprop_momentum,
					epsilon=flags.opt_epsilon)
	elif flags.optimizer == 'sgd'
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	else:
		raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
	
	return optimizer


def retore_meta_graph_and_weights(sess, meta_file_path, checkpoint_path):
	
    if tf.gfile.IsDirectory(checkpoint_path):
	    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

	# load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_file_path)
    saver.restore(sess, checkpoint_path)

    return sess

"""Converting a meta graph file into events file for Tensorboard visualization/

Argss: 
  - meta_file_path
  - checkpoint_path
Returns:
"""
def meta_to_events(meta_file_path, log_path):

    g = tf.Graph()
    with g.as_default():
        tf.train.import_meta_graph(meta_file_path)
	
        with tf.Session() as sess:
		    filewriter = tf.train.FileWriter(log_path, sess.graph)
        filewriter.close()

def get_default_graph():
    return tf.get_default_graph()

def get_tensor_by_name(graph, node_name):
    return graph.get_tensor_by_name(node_name)

def get_init_fn(flags):
    if flags.checkpoint_path is None:
		return None
	# Warn the user if a checkpoint exists in the train_dir. Then ignore.
	if tf.train.latest_checkpoint(flags.train_dir):
		tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in flags.checkpoint_exclude_scopes.split(',')]

    # TODO variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        exclude = False
		for exclusion in exclusions:
				if var.op.name.startswith(exclusion):
					break
		else:
			variables_to_restore.append(var)

	if tf.gfile.IsDirectory(flags.checkpoint_path):
		checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
	else:
		checkpoint_path = flags.checkpoint_path

	tf.logging.info('Fine-tuning from %s. Ignore missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

	return slim.assign_from_checkpoint_fn(
			checkpoint_path,
			variables_to_restore,
			ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
	if flags.trainable_scopes in None:
		return tf.trainable_variables()
	else:
		scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

	variables_to_train = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variables_to_train.extend(variables)
	
	return variables_to_train

def add_variables_summaries(learning_rate):
	summaries = []
	for variable in slim.get_model_variables():
		summaries.append(tf.summary.histogram(variable.op.name, variable))
	summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))

	return summaries
