import sys, os
import tensorflow as tf

sys.path.append(os.path.realpath('../..'))
from book_code.data_utils import *
from book_code.logmanager import *
import math
import getopt

logger.propagate = False

batch_size = 32
num_steps = 100001
learning_rate = 0.1

patch_size = 5
depth_inc = 4
num_hidden_inc = 32
dropout_prob = 0.8

conv_layers = 3
SEED = 11215

stddev = 0.05
stddev_fc = 0.01

regularization_factor = 5e-4

data_showing_step = 50

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
        max_pool = tf.nn.max_pool(relu, ksize=[1, last_pool_kernel_size, last_pool_kernel_size, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    with tf.name_scope('Layer_4') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv4'], strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        bias_add = tf.nn.bias_add(conv, biases['conv4'], name='bias_add_4')
        relu = tf.nn.relu(bias_add, name=scope)

    shape = relu.get_shape().as_list()
    reshape = tf.reshape(relu, [shape[0], shape[1] * shape[2] * shape[3]])

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


dataset, image_size, num_of_classes, num_channels = prepare_cifar_10_dataset()

first_fully_connected_nodes = 512
last_pool_kernel_size = 3
if image_size == 32:
    first_fully_connected_nodes = 512
    last_pool_kernel_size = 3
elif image_size == 28:
    first_fully_connected_nodes = 512
    last_pool_kernel_size = 2

#dataset, image_size, num_of_classes, num_channels = prepare_not_mnist_dataset()
print "Image Size: ", image_size
print "Number of Classes: ", num_of_classes
print "Number of Channels", num_channels
dataset = reformat(dataset, image_size, num_channels, num_of_classes)

print('Training set', dataset.train_dataset.shape, dataset.train_labels.shape)
print('Validation set', dataset.valid_dataset.shape, dataset.valid_labels.shape)
print('Test set', dataset.test_dataset.shape, dataset.test_labels.shape)

#new_valid = (np.array([imresize(x, (227, 227, 3)).astype(float) for x in dataset.valid_dataset]) - 255 / 2) / 255
#print(new_valid)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size, image_size, num_channels), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')
    tf_valid_dataset = tf.constant(dataset.valid_dataset, name='VALID_DATASET')
    tf_test_dataset = tf.constant(dataset.test_dataset, name='TEST_DATASET')
    tf_random_dataset = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels),
                                               name='RANDOM_DATA')
    learning_rate_decayed = tf.placeholder(tf.float32, shape=[], name='learning_rate_decayed')

    print ("Image Size", image_size)
    print ("Conv Layers", conv_layers)
    print ("fc_first_layer_dimen", fc_first_layer_dimen(image_size, conv_layers))

    # Variables.
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([3, 3, num_channels, 128], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv1'),
        'conv2': tf.Variable(tf.truncated_normal([3, 3, 128, 384], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv2'),
        'conv3': tf.Variable(tf.truncated_normal([1, 1, 384, 512], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv3'),
        'conv4': tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv4'),
        'fc6': tf.Variable(tf.truncated_normal([first_fully_connected_nodes, 256], dtype=tf.float32,
                                               stddev=stddev_fc, seed=SEED), name='weights_fc6'),
        'fc7': tf.Variable(tf.truncated_normal([256, num_of_classes], dtype=tf.float32,
                                               stddev=stddev_fc, seed=SEED), name='weights_fc7')
    }
    biases = {
        'conv1': tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases_conv1'),
        'conv2': tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases_conv1'),
        'conv3': tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases_conv1'),
        'conv4': tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases_conv4'),
        'fc6': tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                           trainable=True, name='biases_fc6'),
        'fc7': tf.Variable(tf.constant(0.1, shape=[num_of_classes], dtype=tf.float32),
                           trainable=True, name='biases_fc7'),
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_decayed).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset, weights, biases, TRAIN=False))
    valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases, TRAIN=False))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases, TRAIN=False))
    random_prediction = tf.nn.softmax(nn_model(tf_random_dataset, weights, biases, TRAIN=False))

modelRestoreFile = os.path.realpath('../notMNIST_ann')


def evaluate_cifar_10_image(image_file):
    dictionary = {0: 'airplane',
                  1: 'automobile',
                  2: 'bird',
                  3: 'cat',
                  4: 'deer',
                  5: 'dog',
                  6: 'frog',
                  7: 'horse',
                  8: 'ship',
                  9: 'truck',}

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        print "Restore Session from " + modelRestoreFile
        saver.restore(session, modelRestoreFile)
        print("Model restored from " + modelRestoreFile)
        image = (image_file.astype(float) -
                      255 / 2) / 255
        random_data = np.ndarray((1, image_size, image_size, num_channels), dtype=np.float32)
        random_data[0, :, :, :] = image
        feed_dict = {tf_random_dataset: random_data}
        output = session.run(
            [random_prediction], feed_dict=feed_dict)
        for i, smx in enumerate(output):
            prediction = smx[0].argmax(axis=0)
            print 'The prediction is: %d' % (prediction)

            return dictionary[prediction]

