import sys, os
import tensorflow as tf

sys.path.append(os.path.realpath('../..'))

from book_code.data_utils import *
from book_code.logmanager import *
import math

batch_size = 32
num_steps = 6001
learning_rate = 0.1
num_channels = 1

patch_size = 5
depth_inc = 4
num_hidden_inc = 32
dropout_prob = 0.8

log_location = '/tmp/alex_nn_log'

def reformat(data, image_size, num_channels, num_of_classes):
    data.train_dataset = data.train_dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    data.valid_dataset = data.valid_dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    data.test_dataset = data.test_dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    data.train_labels = (np.arange(num_of_classes) == data.train_labels[:, None]).astype(np.float32)
    data.valid_labels = (np.arange(num_of_classes) == data.valid_labels[:, None]).astype(np.float32)
    data.test_labels = (np.arange(num_of_classes) == data.test_labels[:, None]).astype(np.float32)

    return data


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

#For same padding the output width or height = ceil(width or height / stride) respectively
def fc_first_layer_dimen(image_size, layers):
    output = image_size
    for x in range(layers):
        output = math.ceil(output/2.0)
    return int(output)

def nn_model(data, weights, biases, TRAIN=False):
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)

    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)

    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        relu = tf.nn.relu(bias_add, name='relu_3')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
        if TRAIN:
            max_pool = tf.nn.dropout(max_pool, dropout_prob, name='dropout_1')

    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.name_scope('FC_Layer_1') as scope:
        matmul = tf.matmul(reshape, weights['fc1'], name='fc1_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc1'], name='fc1_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_2') as scope:
        matmul = tf.matmul(relu, weights['fc2'], name='fc2_matmul')
        layer_fc2 = tf.nn.bias_add(matmul, biases['fc2'], name=scope)

    return layer_fc2


not_mnist, image_size, num_of_classes = prepare_not_mnist_dataset()
not_mnist = reformat(not_mnist, image_size, num_channels, num_of_classes)

print('Training set', not_mnist.train_dataset.shape, not_mnist.train_labels.shape)
print('Validation set', not_mnist.valid_dataset.shape, not_mnist.valid_labels.shape)
print('Test set', not_mnist.test_dataset.shape, not_mnist.test_labels.shape)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size, image_size, num_channels), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')
    tf_valid_dataset = tf.constant(not_mnist.valid_dataset, name='VALID_DATASET')
    tf_test_dataset = tf.constant(not_mnist.test_dataset, name='TEST_DATASET')

    first_fully_conntected_size = int(math.ceil(float()))
    # Variables.
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth_inc]), name='weights'),
        'conv2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_inc, depth_inc]), name='weights'),
        'conv3': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_inc, depth_inc]), name='weights'),
        'fc1': tf.Variable(
            tf.truncated_normal([(fc_first_layer_dimen(image_size, 3) ** 2) * depth_inc,
                                 num_hidden_inc]), name='weights'),
        'fc2': tf.Variable(tf.truncated_normal([num_hidden_inc, num_of_classes]), name='weights')
    }
    biases = {
        'conv1': tf.Variable(tf.zeros([depth_inc]), name='biases'),
        'conv2': tf.Variable(tf.zeros([depth_inc]), name='biases'),
        'conv3': tf.Variable(tf.zeros([depth_inc]), name='biases'),
        'fc1': tf.Variable(tf.zeros([num_hidden_inc], name='biases')),
        'fc2': tf.Variable(tf.zeros([num_of_classes], name='biases'))
    }

    for weight_key in sorted(weights.keys()):
        _ = tf.histogram_summary(weight_key + '_weights', weights[weight_key])

    for bias_key in sorted(biases.keys()):
        _ = tf.histogram_summary(bias_key + '_biases', biases[bias_key])

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases, True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    _ = tf.scalar_summary('nn_loss', loss)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases))

with tf.Session(graph=graph) as session:
    # saving graph
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(log_location, session.graph_def)

    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
        sys.stdout.flush()
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (not_mnist.train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = not_mnist.train_dataset[offset:(offset + batch_size), :]
        batch_labels = not_mnist.train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [ optimizer, loss, train_prediction], feed_dict=feed_dict)

        #writer.add_summary(summary_result, step)

        if (step % 500 == 0):
            logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                step, accuracy(predictions, batch_labels), accuracy(
                valid_prediction.eval(), not_mnist.valid_labels), l))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), not_mnist.test_labels))
