import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras import datasets

from tensorflow.contrib.keras import backend as K
assert K.image_data_format() == 'channels_last'


def error_rate_single(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  argmax = np.argmax(predictions, 1)
  return 100.0 - (
      100.0 *
      np.sum(argmax == labels) /
      predictions.shape[0])


def error_rate(predictions_all_steps, labels):
	num_unroll_steps = len(predictions_all_steps)
	return [error_rate_single(predictions_all_steps[i], labels) for i in range(num_unroll_steps)]


class Dataset(object):
    def __init__(self, keras_dataset, num_labels):
        VALIDATION_SIZE = 5000

        self._keras_dataset = keras_dataset
        (self._x_train, self._y_train), (self._x_test, self._y_test) = keras_dataset.load_data()
        self._x_train = self.preprocess(self._x_train)
        self._x_test = self.preprocess(self._x_test)

        if len(self._y_train.shape) == 2:
            assert len(self._y_test.shape) == 2
            assert self._y_train.shape[1] == self._y_test.shape[1] == 1

            self._y_train = self._y_train[:, 0]
            self._y_test = self._y_test[:, 0]

        assert self._x_train.shape[1:] == self._x_test.shape[1:]
        self._image_shape = self._x_train.shape[1:]

        self._x_val = self._x_train[:VALIDATION_SIZE, ...]
        self._y_val = self._y_train[:VALIDATION_SIZE]
        self._x_train = self._x_train[VALIDATION_SIZE:, ...]
        self._y_train = self._y_train[VALIDATION_SIZE:]

        self._num_labels = num_labels

    @staticmethod
    def preprocess(array):
        # rescale from [0, 255] to [-0.5, +0.5]

        if len(array.shape) == 3:
            array = array[:, :, :, np.newaxis]
        array = array.astype(np.float32)
        array = (array - (255.0/2.0)) / 255.0

        return array

    def batch_X_shape(self, batch_size):
        return (batch_size, ) + self._image_shape

    def batch_y_shape(self, batch_size):
        return (batch_size, )

    def get_data(self):
        return (self._x_train, self._y_train), (self._x_val, self._y_val), (self._x_test, self._y_test)

    @property
    def train_size(self):
        return self._y_train.shape[0]

    @property
    def num_labels(self):
        return self._num_labels


def get_dataset(name):
    if name == 'mnist':
        return Dataset(datasets.mnist, num_labels=10)
    if name == 'cifar10':
        return Dataset(datasets.cifar10, num_labels=10)
    else:
        raise ValueError("Don't know {} name!".format(name))


# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess, eval_batch_size,
                    eval_data_placeholder, eval_prediction_tensor):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < eval_batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    # predictions = numpy.ndarray(shape=(NUM_UNROLL_STEPS, size, NUM_LABELS), dtype=numpy.float32)
    predictions = []
    for begin in range(0, size, eval_batch_size):
        end = begin + eval_batch_size
        if end <= size:
            # predictions[:, begin:end, :] = sess.run(
            predictions.append(sess.run(
                eval_prediction_tensor,
                feed_dict={eval_data_placeholder: data[begin:end, ...]})
			)
        else:
            # batch_predictions = sess.run(
            predictions.append(sess.run(
                eval_prediction_tensor,
                feed_dict={eval_data_placeholder: data[-eval_batch_size:, ...]})[:, begin - size:, :]
            )
            # predictions[:, begin:, :] = batch_predictions[:, begin - size:, :]
    ans = np.concatenate(predictions, axis=1)
    # import ipdb; ipdb.set_trace(context=15)
    return ans


def run_train(build_func, train_config, dataset, build_func_kwargs=None):

    optimizer = train_config['optimizer']
    batch_var = train_config['batch_var']
    learning_rate_var = train_config['learning_rate_var']
    train_batch_size = train_config['train_batch_size']
    eval_batch_size = train_config['eval_batch_size']
    num_epochs = train_config['num_epochs']
    eval_frequency = train_config['eval_frequency']

    (train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels) = dataset.get_data()

    train_data_node = tf.placeholder(tf.float32, shape=dataset.batch_X_shape(train_batch_size))
    train_labels_node = tf.placeholder(tf.int64, shape=dataset.batch_y_shape(train_batch_size))
    eval_data = tf.placeholder(tf.float32, shape=dataset.batch_X_shape(eval_batch_size))
    
    # Predictions for the current training minibatch.
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        train_prediction, loss = build_func(train_data_node, training=True, train_labels_node=train_labels_node,
                                            num_labels=dataset.num_labels, **build_func_kwargs)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(loss, global_step=batch_var)
    
    # Predictions for the test and validation, which we'll compute less often.
    with tf.variable_scope("model", reuse=True):
            eval_prediction, _ = build_func(eval_data, training=False, train_labels_node=train_labels_node,
                                            num_labels=dataset.num_labels, **build_func_kwargs)
    
    stdout_lines = []
    
    # Create a local session to run the training.
    start_time = time.time()
    
    with tf.Session() as sess:    
    	# Run all the initializers to prepare the trainable parameters.
    	tf.global_variables_initializer().run()
    	print('Initialized!')
    	
    	# Loop through training steps.
    	for step in range(int(num_epochs * dataset.train_size) // train_batch_size):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * train_batch_size) % (dataset.train_size - train_batch_size)
            batch_data = train_data[offset:(offset + train_batch_size), ...]
            batch_labels = train_labels[offset:(offset + train_batch_size)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
    
            # print some extra information once reach the evaluation frequency
            if step % eval_frequency == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate_var, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
    
                stdout_lines.append('Step %d (epoch %.2f), %.1f ms\n' %
                      (step, float(step) * train_batch_size / dataset.train_size,
                       1000 * elapsed_time / eval_frequency))
                print(stdout_lines[-1].strip())
                stdout_lines.append('Minibatch loss: %.3f, learning rate: %.6f\n' % (l, lr))
                print(stdout_lines[-1].strip())
                stdout_lines.append('Minibatch error: {}\n'.format(error_rate(predictions, batch_labels)))
                print(stdout_lines[-1].strip())
                stdout_lines.append('Validation error: {}\n'
                    .format(error_rate(eval_in_batches(validation_data, sess, eval_batch_size,
                                                       eval_data, eval_prediction),
                                       validation_labels
                            )
                    )
                )
                print(stdout_lines[-1].strip())
                sys.stdout.flush()
                # return stdout_lines
    
        # Finally print the result!
    	test_error = error_rate(eval_in_batches(test_data, sess, eval_batch_size,
                                                eval_data, eval_prediction),
                                test_labels)
    	stdout_lines.append('Test error: {}\n'.format(test_error))
    	print(stdout_lines[-1].strip())

    return stdout_lines

