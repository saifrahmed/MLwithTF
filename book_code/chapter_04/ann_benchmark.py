import sys, os
import tensorflow as tf
sys.path.append(os.path.realpath('../..'))
from book_code.data_utils import *
from book_code.logmanager import *
import math
import getopt
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
stddev = 0.1

data_showing_step = 500

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
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 4, 4, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool1')
        max_pool = tf.nn.lrn(max_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope)

    print "Layer 1 CONV", conv.get_shape()
    #print "Layer 1 RELU", relu.get_shape()
    #print "Layer 1 POOL", max_pool.get_shape()
    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool2')
        max_pool = tf.nn.lrn(max_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope)

    print "Layer 2 CONV", conv.get_shape()
    #print "Layer 2 RELU", relu.get_shape()
    #print "Layer 2 POOL", max_pool.get_shape()
    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        max_pool = tf.nn.relu(bias_add, name='relu_3')
        #max_pool = tf.nn.max_pool(max_pool, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='', name=scope)
        #if TRAIN:
        #    max_pool = tf.nn.dropout(max_pool, dropout_prob, seed=SEED, name='dropout')

    print "Layer 3 CONV", conv.get_shape()
    #print "Layer 3 RELU", relu.get_shape()
    #print "Layer 3 POOL", max_pool.get_shape()

    with tf.name_scope('Layer_4') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv4'], strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        bias_add = tf.nn.bias_add(conv, biases['conv4'], name='bias_add_4')
        max_pool = tf.nn.relu(bias_add, name='relu_4')
        # max_pool = tf.nn.max_pool(max_pool, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='', name=scope)
        # if TRAIN:
        #    max_pool = tf.nn.dropout(max_pool, dropout_prob, seed=SEED, name='dropout')

    print "Layer 4 CONV", conv.get_shape()
    # print "Layer 3 RELU", relu.get_shape()
    # print "Layer 3 POOL", max_pool.get_shape()

    with tf.name_scope('Layer_5') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv5'], strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        bias_add = tf.nn.bias_add(conv, biases['conv5'], name='bias_add_5')
        max_pool = tf.nn.relu(bias_add, name='relu_5')
        max_pool = tf.nn.max_pool(max_pool, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)
        # if TRAIN:
        #    max_pool = tf.nn.dropout(max_pool, dropout_prob, seed=SEED, name='dropout')

    print "Layer 5 CONV", conv.get_shape()
    # print "Layer 3 RELU", relu.get_shape()
    # print "Layer 3 POOL", max_pool.get_shape()

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
        layer_fc8 = tf.nn.bias_add(matmul, biases['fc8'], name=scope)

    return layer_fc8


dataset, image_size, num_of_classes, num_channels = prepare_cifar_10_dataset()
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
                                      shape=(batch_size, 227, 227, num_channels), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')
    tf_valid_dataset = tf.constant(dataset.valid_dataset, name='VALID_DATASET')
    tf_test_dataset = tf.constant(dataset.test_dataset, name='TEST_DATASET')
    tf_random_dataset = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels),
                                               name='RANDOM_DATA')

    print ("Image Size", image_size)
    print ("Conv Layers", conv_layers)
    print ("fc_first_layer_dimen", fc_first_layer_dimen(image_size, conv_layers))

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
        'fc6': tf.Variable(tf.truncated_normal([9216, 4096],
                                               stddev=stddev, seed=SEED), name='weights'),
        'fc7': tf.Variable(tf.truncated_normal([4096, 4096],
                                               stddev=stddev, seed=SEED), name='weights'),
        'fc8': tf.Variable(tf.truncated_normal([4096, num_of_classes],
                                               stddev=stddev, seed=SEED), name='weights')
    }
    biases = {
        'conv1': tf.Variable(tf.zeros([96]), name='biases'),
        'conv2': tf.Variable(tf.zeros([256]), name='biases'),
        'conv3': tf.Variable(tf.zeros([384]), name='biases'),
        'conv4': tf.Variable(tf.zeros([384]), name='biases'),
        'conv5': tf.Variable(tf.zeros([256]), name='biases'),
        'fc6': tf.Variable(tf.zeros([4096], name='biases')),
        'fc7': tf.Variable(tf.zeros([4096], name='biases')),
        'fc8': tf.Variable(tf.zeros([num_of_classes], name='biases'))
    }

    for weight_key in sorted(weights.keys()):
        _ = tf.histogram_summary(weight_key + '_weights', weights[weight_key])

    for bias_key in sorted(biases.keys()):
        _ = tf.histogram_summary(bias_key + '_biases', biases[bias_key])

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases, True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(weights['fc6']) + tf.nn.l2_loss(biases['fc6']) +
                    tf.nn.l2_loss(weights['fc7']) + tf.nn.l2_loss(biases['fc7']) +
                    tf.nn.l2_loss(weights['fc8']) + tf.nn.l2_loss(biases['fc8']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    _ = tf.scalar_summary('nn_loss', loss)

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases))
    #test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases))
    #random_prediction = tf.nn.softmax(nn_model(tf_random_dataset, weights, biases))

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
            sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
            sys.stdout.flush()
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (dataset.train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = dataset.train_dataset[offset:(offset + batch_size), :]
            batch_data = (np.array([imresize(x, (227, 227, 3)).astype(float) for x in batch_data]) - 255 / 2) / 255
            batch_labels = dataset.train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            #print feed_dict
            summary_result, _, l, predictions = session.run(
                [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)

            writer.add_summary(summary_result, step)

            if (step % data_showing_step == 0):
                #logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                #    step, accuracy(predictions, batch_labels), accuracy(
                #    valid_prediction.eval(), dataset.valid_labels), l))
                logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                    step, accuracy(predictions, batch_labels), -1.0, l))
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