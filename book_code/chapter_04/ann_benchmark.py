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


dataset, image_size, num_of_classes, num_channels = prepare_not_mnist_dataset()

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

#modelRestoreFile = os.path.realpath('../notMNIST_ann')
modelRestoreFile = None
modelSaveFile = os.path.realpath('../notMNIST_ann')
#evaluateFile = '/home/shams/Desktop/test_images_2/MDEtMDEtMDAudHRm.png'
evaluateFile = None

try:
    opts, args = getopt.getopt(sys.argv[1:],"ur:s:e:",["modelRestoreFile=","modelSaveFile=","evaluateFile="])
except getopt.GetoptError:
    print 'ann_benchmark.py -r <path to model file to restore from>'
    print 'ann_benchmark.py -s <destination to persist model file to>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-u':
        print 'ann_benchmark usage:'
        print 'ann_benchmark.py -r <path to model file to restore from>'
        print 'ann_benchmark.py -s <destination to persist model file to>'
        sys.exit()
    elif opt in ("-r", "--modelRestoreFile"):
        modelRestoreFile = arg
    elif opt in ("-s", "--modelSaveFile"):
        modelSaveFile = arg
    elif opt in ("-e", "--evaluateFile"):
        evaluateFile = arg

print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
if (modelRestoreFile is not None):
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        print "Restore Session from " + modelRestoreFile
        saver.restore(session, modelRestoreFile)
        print("Model restored from " + modelRestoreFile)

        print test_prediction
        print test_prediction.eval().shape
        print dataset.test_labels.shape

        for i,smx in enumerate(test_prediction.eval()):
            actual=dataset.test_labels[i].argmax(axis=0)
            predicted=smx.argmax(axis=0)
            print i, "Actual", actual, "Prediction", predicted, "Correct" if actual==predicted else "Incorrect"
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.test_labels))

else:
    print "Run NEW session"
    with tf.Session(graph=graph) as session:
        # saving graph
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(log_location, session.graph_def)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        print("Initialized")

        for step in range(num_steps):

            if 30000 < step < 80000:
                learning_rate = 0.02 / 5
            elif 80000 <= step < 200000:
                learning_rate = 0.02 / 10
            else:
                learning_rate = 0.02

            sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
            sys.stdout.flush()
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (dataset.train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = dataset.train_dataset[offset:(offset + batch_size), :]
            batch_labels = dataset.train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels,
                         learning_rate_decayed: learning_rate}
            #print feed_dict
            summary_result, _, l, predictions = session.run(
                [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)

            writer.add_summary(summary_result, step)

            if (step % data_showing_step == 0):
                logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f Learning Rate: %f' % (
                    step, accuracy(predictions, batch_labels), accuracy(
                    valid_prediction.eval(), dataset.valid_labels), l,
                   learning_rate))
                #logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                #    step, accuracy(predictions, batch_labels), -1.0, l))
        if (modelSaveFile is not None):
            save_path = saver.save(session, modelSaveFile)
            print("Model saved in file: %s" % save_path)
        else:
            print("Trained Model discarded, no save details provided")
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.test_labels))

if (evaluateFile is not None):
    print "We wish to evaluate the file " + evaluateFile

    if (modelRestoreFile is not None):
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            print "Restore Session from " + modelRestoreFile
            saver.restore(session, modelRestoreFile)
            print("Model restored from " + modelRestoreFile)

            image = (ndimage.imread(evaluateFile).astype(float) -
                          255 / 2) / 255
            image = image.reshape((image_size, image_size, num_channels)).astype(np.float32)
            random_data = np.ndarray((1, image_size, image_size, num_channels), dtype=np.float32)
            random_data[0, :, :, :] = image

            feed_dict = {tf_random_dataset: random_data}
            output = session.run(
                [random_prediction], feed_dict=feed_dict)

            for i, smx in enumerate(output):
                prediction = smx[0].argmax(axis=0)
                print 'The prediction is: %d' % (prediction)