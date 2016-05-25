import sys
import numpy as np
import pickle
from scipy.misc import imsave, imresize
import tensorflow as tf

cifar_data_folder = '../../datasets/CIFAR-10/data/cifar-10-batches-py'
train_pickle_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                      'data_batch_5']
test_pickle_files = ['test_batch']

test_images_folder = '/home/shams/Desktop/pycharm_projects/cifar-10_improv/test_images'
current_pickle_file = cifar_data_folder + '/' + train_pickle_files[0]

log_location = '/home/shams/Desktop/pycharm_projects/cifar-10_improv/alex_nn_log'
batch_size = 32
learning_rate = 0.02
num_of_classes = 10
SEED = 11215
stddev = 0.1
stddev_fc = 0.05
data_showing_step = 50
num_steps = 100001
regularization_factor = 0   #5e-4


def load_cifar_10_pickle(pickle_file):
    fo = open(pickle_file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return np.array(dict['data']).astype(float), np.array(dict['labels'])

def load_cifar_10_from_pickles(train_pickle_files, test_pickle_files, pickle_batch_size, image_size,
                                   num_of_channels):

    all_train_data = np.ndarray(shape=(pickle_batch_size * len(train_pickle_files),
                                       image_size * image_size * num_of_channels),
                                dtype=np.float32)

    all_train_labels = np.ndarray(shape=pickle_batch_size * len(train_pickle_files), dtype=np.float32)

    all_test_data = np.ndarray(shape=(pickle_batch_size * len(test_pickle_files),
                                      image_size * image_size * num_of_channels),
                               dtype=np.float32)
    all_test_labels = np.ndarray(shape=pickle_batch_size * len(test_pickle_files), dtype=np.float32)

    print('Started loading training data')
    for index, train_pickle_file in enumerate(train_pickle_files):
        all_train_data[index * pickle_batch_size: (index + 1) * pickle_batch_size, :], \
        all_train_labels[index * pickle_batch_size: (index + 1) * pickle_batch_size] = \
            load_cifar_10_pickle(train_pickle_file)
    print('Finished loading training data\n')

    print('Started loading testing data')
    for index, test_pickle_file in enumerate(test_pickle_files):
        all_test_data[index * pickle_batch_size: (index + 1) * pickle_batch_size, :], \
        all_test_labels[index * pickle_batch_size: (index + 1) * pickle_batch_size] = \
            load_cifar_10_pickle(test_pickle_file)
    print('Finished loading testing data')

    return all_train_data, all_train_labels, all_test_data, all_test_labels


def reshape_linear(data, shape):
    output = np.ndarray(shape=(data.shape[0], shape, shape, 3), dtype=np.float32)
    for index, image in enumerate(data):
        sys.stdout.write('Reshaping image %d of %d                    \r' % (index + 1, data.shape[0]))
        sys.stdout.flush()
        for row in range(shape):
            for col in range(shape):
                image_index = row * shape + col
                output[index, row, col, :] = [image[image_index],
                                              image[shape**2 + image_index],
                                              image[(shape**2)*2 + image_index]]
    return output

def reshape_linear_fast(data, shape, channels):
    linear_pixels = shape**2
    sequence = np.ndarray(shape=(linear_pixels*channels), dtype=np.int32)
    for i in range(linear_pixels):
        start = i * (channels)
        end = (i+1) * (channels)
        sequence[start:end] = range(i, linear_pixels*channels, linear_pixels)
    return data[:, sequence].reshape(-1, shape, shape, channels)


def reshape_labels(data, classes):
    return (np.arange(classes) == data[:, None]).astype(np.float32)

def normalize(data):
    return (data - 255/2) / 255


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def nn_model(data, weights, biases, TRAIN=False):
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 2, 2, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        lrn = tf.nn.lrn(relu, 5, bias=1.0, alpha=0.0001, beta=0.75, name='lrn1')
        max_pool = tf.nn.max_pool(lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        lrn = tf.nn.lrn(relu, 5, bias=1.0, alpha=0.0001, beta=0.75, name='lrn2')
        max_pool = tf.nn.max_pool(lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        relu = tf.nn.relu(bias_add, name='relu_3')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.name_scope('FC_Layer_6') as scope:
        matmul = tf.matmul(reshape, weights['fc6'], name='fc6_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc6'], name='fc6_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)
        if(TRAIN):
            relu = tf.nn.dropout(relu, 0.5, seed=SEED, name='dropout_fc6')

    with tf.name_scope('FC_Layer_7') as scope:
        matmul = tf.matmul(relu, weights['fc7'], name='fc7_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc7'], name='fc7_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    return relu

data, labels, test_data, test_labels = load_cifar_10_from_pickles([cifar_data_folder + '/' + x for x in
                                                                   train_pickle_files],
                                                                  [cifar_data_folder + '/' + x for x in
                                                                   test_pickle_files], 10000, 32, 3)

train_data = normalize(reshape_linear_fast(data[0:45000], 32, 3))
train_labels = reshape_labels(labels[0:45000], 10)
valid_data = normalize(reshape_linear_fast(data[45000:50000], 32, 3))
valid_labels = reshape_labels(labels[45000:50000], 10)
test_data = normalize(reshape_linear_fast(test_data, 32, 3))
test_labels = reshape_labels(test_labels, 10)

#for i in range(1000):
#    imsave(test_images_folder + '/image_train' + str(i) + '.png', train_data[i])
#    imsave(test_images_folder + '/image_valid' + str(i) + '.png', valid_data[i])
#    imsave(test_images_folder + '/image_test' + str(i) + '.png', test_data[i])


print('saved')

print(train_data.shape, train_labels.shape, valid_data.shape, valid_labels.shape, test_data.shape, test_labels.shape)

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, 32, 32, 3), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')
    tf_valid_dataset = tf.constant(valid_data, tf.float32,
                                      shape=(valid_data.shape[0], 32, 32, 3), name='VALID_DATASET')
    tf_test_dataset = tf.constant(test_data, tf.float32,
                                      shape=(test_data.shape[0], 32, 32, 3), name='TEST_DATASET')

    # Variables.
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv2': tf.Variable(tf.truncated_normal([3, 3, 64, 194], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv3': tf.Variable(tf.truncated_normal([1, 1, 194, 256], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights'),
        'fc6': tf.Variable(tf.truncated_normal([256, 128], dtype=tf.float32,
                                               stddev=stddev_fc, seed=SEED), name='weights'),
        'fc7': tf.Variable(tf.truncated_normal([128, num_of_classes], dtype=tf.float32,
                                               stddev=stddev_fc, seed=SEED), name='weights')
    }
    biases = {
        'conv1': tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases'),
        'conv2': tf.Variable(tf.constant(0.1, shape=[194], dtype=tf.float32),
                         trainable=True, name='biases'),
        'conv3': tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases'),
        'fc6': tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases'),
        'fc7': tf.Variable(tf.constant(0.1, shape=[num_of_classes], dtype=tf.float32),
                         trainable=True, name='biases'),
    }

    for weight_key in sorted(weights.keys()):
        _ = tf.histogram_summary(weight_key + '_weights', weights[weight_key])

    for bias_key in sorted(biases.keys()):
        _ = tf.histogram_summary(bias_key + '_biases', biases[bias_key])

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases, TRAIN=True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(weights['fc6']) + tf.nn.l2_loss(biases['fc6']) +
                    tf.nn.l2_loss(weights['fc7']) + tf.nn.l2_loss(biases['fc7']))
    # Add the regularization term to the loss.
    loss += regularization_factor * regularizers

    _ = tf.scalar_summary('nn_loss', loss)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset, weights, biases, TRAIN=False))
    valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases, TRAIN=False))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases, TRAIN=False))

with tf.Session(graph=graph) as session:
    # saving graph
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(log_location, session.graph_def)

    tf.initialize_all_variables().run()

    print("Initialized")
    for step in range(num_steps):
        sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
        sys.stdout.flush()
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        #print(np.argmax(batch_labels, 1))
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        # print feed_dict
        summary_result, _, l, predictions = session.run(
            [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)

        writer.add_summary(summary_result, step)

        if (step % data_showing_step == 0):
            print('Step %03d  Acc-Minibatch: %03.2f%% Acc-Valid: %03.2f%% Minibatch loss %f' %
                  (step, accuracy(predictions, batch_labels), accuracy(valid_prediction.eval(), valid_labels), l))

    print('Acc-Test: %03.2f%%' % accuracy(test_prediction.eval(), test_labels))