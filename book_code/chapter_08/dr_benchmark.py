import os
import sys

import tensorflow as tf

sys.path.append(os.path.realpath('../..'))
from book_code.data_utils import *
from book_code.logmanager import *
from scipy.misc import imresize

logger.propagate = False

batch_size = 32
num_steps = 3001
learning_rate = 0.1

patch_size = 5
depth_inc = 4
num_hidden_inc = 32
dropout_prob = 0.8

conv_layers = 3
SEED = 11215
stddev = 0.1/50.0
stddev_fc = 0.005/50.0
data_showing_step = 5

log_location = '/tmp/alex_nn_log'


def reformat(dataset, labels, image_shape, num_labels, num_channels):
    dataset = dataset.reshape(
        (-1, image_shape[0], image_shape[1], num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def get_valid_and_test_batch_size(image_shape, safety_percentage, train_batch_size,
                                  valid_size, test_size, memory_available):
    image_bytes = image_shape[0] * image_shape[1] * image_shape[2] * 4
    train_batch_bytes = train_batch_size * image_bytes
    remaining_bytes = int(memory_available * safety_percentage - train_batch_bytes)

    valid_total_bytes = valid_size * image_bytes
    test_total_bytes = test_size * image_bytes

    if valid_total_bytes >= test_total_bytes:
        if valid_total_bytes <= remaining_bytes // 2:
            valid_batch_size = valid_size
            test_batch_size = test_size
        else:
            if test_total_bytes <= remaining_bytes // 2:
                test_batch_size = test_size

                remaining_valid_bytes = remaining_bytes - test_total_bytes
                valid_batch_size = remaining_valid_bytes // image_bytes
                if valid_batch_size > valid_size:
                    valid_batch_size = valid_size
            else:
                test_batch_size = (remaining_bytes // 2) // image_bytes
                if test_batch_size > test_size:
                    test_batch_size = test_size

                valid_batch_size = (remaining_bytes // 2) // image_bytes
                if valid_batch_size > valid_size:
                    valid_batch_size = valid_size

    else:
        if test_total_bytes <= remaining_bytes // 2:
            test_batch_size = test_size
            valid_batch_size = valid_size
        else:
            if valid_total_bytes <= remaining_bytes // 2:
                valid_batch_size = valid_size

                remaining_test_bytes = remaining_bytes - valid_total_bytes
                test_batch_size = remaining_test_bytes // image_bytes
                if test_batch_size > test_size:
                    test_batch_size = test_size
            else:
                valid_batch_size = (remaining_bytes // 2) // image_bytes
                if valid_batch_size > valid_size:
                    valid_batch_size = valid_size

                test_batch_size = (remaining_bytes // 2) // image_bytes
                if test_batch_size > test_size:
                    test_batch_size = test_size

    return valid_batch_size, test_batch_size


def preprocess_images(image_path, reduced_image_size, num_channels, pixel_depth):

    original_image = ndimage.imread(image_path)
    image_shape = original_image.shape

    if image_shape[0] == image_shape[1]:
        processed_image = original_image
    elif image_shape[0] > image_shape[1]:
        diff = image_shape[0] - image_shape[1]
        if diff % 2 == 0:
            padding = int(diff / 2)
            npad = ((0,0), (padding,padding), (0,0))
            processed_image = np.pad(original_image, pad_width=npad, mode='constant', constant_values=0)
        else:
            padding = int(diff // 2)
            npad = ((0,0), (padding+1,padding), (0,0))
            processed_image = np.pad(original_image, pad_width=npad, mode='constant', constant_values=0)
    else:
        diff = image_shape[1] - image_shape[0]
        if diff % 2 == 0:
            padding = int(diff / 2)
            npad = ((padding,padding), (0,0), (0,0))
            processed_image = np.pad(original_image, pad_width=npad, mode='constant', constant_values=0)
        else:
            padding = int(diff // 2)
            npad = ((padding+1,padding), (0,0), (0,0))
            processed_image = np.pad(original_image, pad_width=npad, mode='constant', constant_values=0)

    processed_image = imresize(processed_image, (reduced_image_size[0], reduced_image_size[1], num_channels),
                               interp='bicubic').astype(float)
    processed_image = (processed_image - 255 / 2) / 255

    return processed_image


def load_batch(dataset_file_paths, labels, offset, batch_size, image_shape, pixel_depth, num_labels,
               num_channels):
    batch_data = np.ndarray(shape=(batch_size, image_shape[0], image_shape[1], num_channels), dtype=np.float32)

    batch_labels = np.ndarray(shape=batch_size, dtype=np.int32)

    image_index = 0
    skipped_images = 0
    index = 0
    length_dataset = len(dataset_file_paths)

    last_seen_good_image = np.array([])
    last_seen_good_label = -1

    while image_index < batch_size and (offset + index) < length_dataset:
        try:
            image = ndimage.imread(dataset_file_paths[offset + index]).astype(float)
            image_data = preprocess_images(image, image_shape, num_channels, pixel_depth)

            last_seen_good_image = image_data
            last_seen_good_label = labels[offset + index]

            if image_data.shape != (image_shape[0], image_shape[1], num_channels):
                print('Unexpected image shape: %s' % str(image_data.shape))
                if last_seen_good_label != -1:
                    batch_data[image_index, :, :, :] = last_seen_good_image
                    batch_labels[image_index] = last_seen_good_label
                    image_index += 1

                skipped_images += 1

            else:
                batch_data[image_index, :, :, :] = image_data
                batch_labels[image_index] = labels[offset + index]
                image_index += 1
        except IOError as e:
            logger.warn('Skipping unreadable image:' + dataset_file_paths[offset + index] + ': ' + str(e))
            if last_seen_good_label != -1:
                batch_data[image_index, :, :, :] = last_seen_good_image
                batch_labels[image_index] = last_seen_good_label
                image_index += 1

            skipped_images += 1

        index += 1

    batch_data = batch_data[0:image_index, :, :, :]
    batch_labels = batch_labels[0:image_index]

    batch_data, batch_labels = reformat(batch_data, batch_labels, image_shape, num_labels, num_channels)

    return batch_data, batch_labels, skipped_images


def accuracy_batches(session, tf_dataset, prediction, dataset,
                     labels, batch_size, image_shape, image_depth,
                     num_classes, channels):
    steps = len(labels) // batch_size
    skipped_images = 0
    total_checked = 0
    correct_sum = 0
    for batch_step in range(steps):
        offset = (batch_step * batch_size)
        batch_data, batch_labels, skipped = load_batch(dataset,
                                                       labels,
                                                       offset + skipped_images,
                                                       batch_size,
                                                       image_shape,
                                                       image_depth,
                                                       num_classes,
                                                       channels)

        skipped_images += skipped

        total_checked += len(batch_labels)
        feed_dict = {tf_dataset: batch_data}
        correct_sum += np.sum(
            np.argmax(session.run(prediction, feed_dict=feed_dict), 1)
            == np.argmax(batch_labels, 1))
        return 100.0 * correct_sum / total_checked


def nn_model(data, weights, biases, TRAIN=False):
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 4, 4, 1], padding='SAME', name='conv1')
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
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('Layer_4') as scope:
        conv = tf.nn.conv2d(relu, weights['conv4'], strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        bias_add = tf.nn.bias_add(conv, biases['conv4'], name='bias_add_4')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('Layer_5') as scope:
        conv = tf.nn.conv2d(relu, weights['conv5'], strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        bias_add = tf.nn.bias_add(conv, biases['conv5'], name='bias_add_5')
        relu = tf.nn.relu(bias_add, name='relu_5')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.name_scope('FC_Layer_6') as scope:
        matmul = tf.matmul(reshape, weights['fc6'], name='fc6_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc6'], name='fc6_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)
        if (TRAIN):
            relu = tf.nn.dropout(relu, 0.5, seed=SEED, name='dropout_fc6')

    with tf.name_scope('FC_Layer_7') as scope:
        matmul = tf.matmul(relu, weights['fc7'], name='fc7_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc7'], name='fc7_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)
        if (TRAIN):
            relu = tf.nn.dropout(relu, 0.5, seed=SEED, name='dropout_fc7')

    with tf.name_scope('FC_Layer_8') as scope:
        matmul = tf.matmul(relu, weights['fc8'], name='fc8_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc8'], name='fc8_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)
        if (TRAIN):
            relu = tf.nn.dropout(relu, 0.5, seed=SEED, name='dropout_fc8')

    with tf.name_scope('FC_Layer_9') as scope:
        matmul = tf.matmul(relu, weights['fc9'], name='fc9_matmul')
        layer_fc9 = tf.nn.bias_add(matmul, biases['fc9'], name=scope)

    return layer_fc9


dataset, image_size, num_of_classes = prepare_DR_dataset()
valid_batch_size, test_batch_size = get_valid_and_test_batch_size((image_size[0], image_size[1], 3), 0.8, batch_size,
                                                                  len(dataset.valid_dataset),
                                                                  len(dataset.test_dataset),
                                                                  2 * (1024 ** 3))

graph = tf.Graph()
with graph.as_default():
    for d in ['/gpu:0', '/gpu:1', '/gpu:2']:
        with tf.device(d):
            tf_train_dataset = tf.placeholder(tf.float32,
                                              shape=(batch_size, image_size[0], image_size[1], 3), name='TRAIN_DATASET')
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')
            tf_valid_dataset = tf.placeholder(tf.float32,
                                              shape=(valid_batch_size, image_size[0], image_size[1], 3),
                                              name='VALID_DATASET')
            tf_test_dataset = tf.placeholder(tf.float32,
                                             shape=(test_batch_size, image_size[0], image_size[1], 3),
                                             name='TEST_DATASET')

            # Variables.
            weights = {
                'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                         stddev=stddev, seed=SEED), name='weights'),
                'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                         stddev=stddev, seed=SEED), name='weights'),
                'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32,
                                                         stddev=stddev, seed=SEED), name='weights'),
                'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32,
                                                         stddev=stddev, seed=SEED), name='weights'),
                'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                                         stddev=stddev, seed=SEED), name='weights'),
                'fc6': tf.Variable(tf.truncated_normal([9216, 4096], dtype=tf.float32,
                                                       stddev=stddev_fc, seed=SEED), name='weights'),
                'fc7': tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32,
                                                       stddev=stddev_fc, seed=SEED), name='weights'),
                'fc8': tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32,
                                                       stddev=stddev, seed=SEED), name='weights'),
                'fc9': tf.Variable(tf.truncated_normal([1000, num_of_classes], dtype=tf.float32,
                                                       stddev=stddev, seed=SEED), name='weights')
            }
            biases = {
                'conv1': tf.Variable(tf.constant(0.1, shape=[96], dtype=tf.float32),
                                     trainable=True, name='biases'),
                'conv2': tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases'),
                'conv3': tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                                     trainable=True, name='biases'),
                'conv4': tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                                     trainable=True, name='biases'),
                'conv5': tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases'),
                'fc6': tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases'),
                'fc7': tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases'),
                'fc8': tf.Variable(tf.constant(0.1, shape=[1000], dtype=tf.float32),
                                   trainable=True, name='biases'),
                'fc9': tf.Variable(tf.constant(0.1, shape=[num_of_classes], dtype=tf.float32),
                                   trainable=True, name='biases')
            }

            # for weight_key in sorted(weights.keys()):
            #    _ = tf.histogram_summary(weight_key + '_weights', weights[weight_key])

            # for bias_key in sorted(biases.keys()):
            #    _ = tf.histogram_summary(bias_key + '_biases', biases[bias_key])

            # Training computation.
            logits = nn_model(tf_train_dataset, weights, biases, TRAIN=True)
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
            train_prediction = tf.nn.softmax(nn_model(tf_train_dataset, weights, biases, TRAIN=False))
            valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases, TRAIN=False))
            test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases, TRAIN=False))

skipped_images = 0

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(graph=graph, config=config) as session:
    # saving graph
    # merged = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter(log_location, session.graph_def)

    tf.initialize_all_variables().run()
    logger.info('Session Initialized')
    for step in range(num_steps):
        sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
        sys.stdout.flush()

        offset = (step * batch_size) % (dataset.train_dataset.shape[0] - batch_size)

        batch_data, batch_labels, skipped = load_batch(dataset.train_dataset, dataset.train_labels,
                                                       offset + skipped_images,
                                                       batch_size,
                                                       (image_size[0], image_size[1]),
                                                       255,
                                                       5, 3)
        skipped_images += skipped
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % data_showing_step == 0:
            # writer.add_summary(sum_string, step)

            # print('Minibatch loss at step %d: %f' % (step, l))
            # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            miniBatchAccuracy = accuracy(predictions, batch_labels)

            validationAccuracy = accuracy_batches(session, tf_valid_dataset, valid_prediction, dataset.valid_dataset,
                                                  dataset.valid_labels, valid_batch_size,
                                                  (image_size[0], image_size[1]), 255, 5, 3)

            # print('Validation accuracy: %.1f%%' % (100.0 * valid_correct_sum / total_valid_checked))
            # print('Step %03d  Acc Minibatch: %03.2f%% \t Acc Val: %03.2f%% \t Minibatch loss %f' % (step, miniBatchAccuracy, validationAccuracy, l))
            logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                step, miniBatchAccuracy, validationAccuracy, l))

    # ---- Calculating test accuracy
    testAccuracy = accuracy_batches(session, tf_test_dataset, test_prediction, dataset.test_dataset,
                                    dataset.test_labels, test_batch_size, (image_size[0],
                                                                           image_size[1]), 255, 5, 3)

    # print('Test accuracy: %.1f%%' % (100.0 * test_correct_sum / total_test_checked))
    logger.info('Test accuracy: %.1f%%' % testAccuracy)
