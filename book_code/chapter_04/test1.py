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
learning_rate = 0.1
num_of_classes = 10
SEED = 11215
stddev = 0.1
data_showing_step = 50
num_steps = 3001


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


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def nn_model(data, weights, biases):
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 4, 4, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool1')
        max_pool = tf.nn.lrn(max_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope)

    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool2')
        max_pool = tf.nn.lrn(max_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope)

    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        max_pool = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('Layer_4') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv4'], strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        bias_add = tf.nn.bias_add(conv, biases['conv4'], name='bias_add_4')
        max_pool = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('Layer_5') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv5'], strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        bias_add = tf.nn.bias_add(conv, biases['conv5'], name='bias_add_5')
        max_pool = tf.nn.relu(bias_add, name='relu_5')
        max_pool = tf.nn.max_pool(max_pool, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.name_scope('FC_Layer_6') as scope:
        matmul = tf.matmul(reshape, weights['fc6'], name='fc6_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc6'], name='fc6_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_7') as scope:
        matmul = tf.matmul(relu, weights['fc7'], name='fc7_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc7'], name='fc7_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_8') as scope:
        matmul = tf.matmul(relu, weights['fc8'], name='fc8_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc8'], name='fc8_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_9') as scope:
        matmul = tf.matmul(relu, weights['fc9'], name='fc9_matmul')
        layer_fc9 = tf.nn.bias_add(matmul, biases['fc9'], name=scope)

    return layer_fc9

data, labels, test_data, test_labels = load_cifar_10_from_pickles([cifar_data_folder + '/' + x for x in
                                                                   train_pickle_files],
                                                                  [cifar_data_folder + '/' + x for x in
                                                                   test_pickle_files], 10000, 32, 3)

train_data = data[0:45000]
train_labels = labels[0:45000]
valid_data = data[45000:50000]
valid_labels = labels[45000:50000]

print(train_data.shape, train_labels.shape, valid_data.shape, valid_labels.shape, test_data.shape, test_labels.shape)

#for i in range(20):
#    reshapedImage = data[i].reshape((3, 32, 32))
#    types = ['nearest', 'bilinear', 'bicubic', 'cubic']
#    for type in types:
#        resizedImage = imresize(reshapedImage, (227, 227, 3), interp=type)
#        imsave(test_images_folder + '/image' + str(i) + '_' + type + '.png', resizedImage)
#        reformatedImage = resizedImage.astype(float)
#    print(i)

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, 227, 227, 3), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')

    # Variables.
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96],
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256],
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                                 stddev=stddev, seed=SEED), name='weights'),
        'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 stddev=stddev, seed=SEED), name='weights'),
        'fc6': tf.Variable(tf.truncated_normal([9216, 128],
                                               stddev=stddev, seed=SEED), name='weights'),
        'fc7': tf.Variable(tf.truncated_normal([128, 64],
                                               stddev=stddev, seed=SEED), name='weights'),
        'fc8': tf.Variable(tf.truncated_normal([64, 32],
                                               stddev=stddev, seed=SEED), name='weights'),
        'fc9': tf.Variable(tf.truncated_normal([32, num_of_classes],
                                               stddev=stddev, seed=SEED), name='weights')
    }
    biases = {
        'conv1': tf.Variable(tf.zeros([96]), name='biases'),
        'conv2': tf.Variable(tf.zeros([256]), name='biases'),
        'conv3': tf.Variable(tf.zeros([384]), name='biases'),
        'conv4': tf.Variable(tf.zeros([384]), name='biases'),
        'conv5': tf.Variable(tf.zeros([256]), name='biases'),
        'fc6': tf.Variable(tf.zeros([128], name='biases')),
        'fc7': tf.Variable(tf.zeros([64], name='biases')),
        'fc8': tf.Variable(tf.zeros([32], name='biases')),
        'fc9': tf.Variable(tf.zeros([num_of_classes], name='biases'))
    }

    for weight_key in sorted(weights.keys()):
        _ = tf.histogram_summary(weight_key + '_weights', weights[weight_key])

    for bias_key in sorted(biases.keys()):
        _ = tf.histogram_summary(bias_key + '_biases', biases[bias_key])

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(weights['fc6']) + tf.nn.l2_loss(biases['fc6']) +
                    tf.nn.l2_loss(weights['fc7']) + tf.nn.l2_loss(biases['fc7']) +
                    tf.nn.l2_loss(weights['fc8']) + tf.nn.l2_loss(biases['fc8']) +
                    tf.nn.l2_loss(weights['fc9']) + tf.nn.l2_loss(biases['fc9']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    _ = tf.scalar_summary('nn_loss', loss)

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

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
        reshapedImages = batch_data.reshape((-1, 3, 32, 32))
        resizedImages = np.array([imresize(reshapedImage, (227, 227, 3), interp='bicubic').astype(np.float32)
                         for reshapedImage in reshapedImages])
        #for index, resizedImage in enumerate(resizedImages):
        #    imsave(test_images_folder + '/image' + str(index) + '_bicubic.png', resizedImage)
        batch_data = (resizedImages - 255/2) / 255
        batch_labels = (np.arange(num_of_classes) == train_labels[offset:(offset + batch_size)][:, None]).astype(np.float32)
        #print(np.argmax(batch_labels, 1))
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        # print feed_dict
        summary_result, _, l, predictions = session.run(
            [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)

        writer.add_summary(summary_result, step)

        if (step % data_showing_step == 0):
            print('Step %03d  Acc Minibatch: %03.2f%% Minibatch loss %f' %
                  (step, accuracy(predictions, batch_labels), l))