import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from logmanager import *

np.random.seed(133)
max_graph_size = 2 * (1024 ** 3)  # (1024 ** 3) means a GB


class Base(object):
    image_depth = -1

    train_classes = np.array([])
    train_classes_labels = np.array([])
    train_image_shape = (-1, -1)
    total_training_classes = -1
    training_classes_samples = np.array([])

    test_classes = np.array([])
    test_classes_labels = np.array([])
    test_image_shape = (-1, -1)
    total_testing_classes = -1
    testing_classes_samples = np.array([])

    train_dataset = np.array([])
    train_labels = np.array([])
    valid_dataset = np.array([])
    valid_labels = np.array([])
    test_dataset = np.array([])
    test_labels = np.array([])

    def __init__(self, safety_percentage, image_depth, train_root=None, test_root=None):
        self.safety_percentage = safety_percentage
        Base.image_depth = image_depth
        if len(Base.train_classes) == 0 and train_root and test_root:
            logger.info("Reviewing training data")
            Base.train_image_shape, Base.total_training_classes, \
            Base.training_classes_samples, Base.train_classes, Base.train_classes_labels = Base.load_classes(train_root)
            logger.info("Reviewing test data")
            Base.test_image_shape, Base.total_testing_classes, \
            Base.testing_classes_samples, Base.test_classes, Base.test_classes_labels = Base.load_classes(test_root)
            logger.info("Done Reviewing data")

            logger.info('Train metadata --> ' +
                        ' ImageShape:' + str(Base.train_image_shape) +
                        ' Classes:' + str(Base.total_training_classes) +
                        ' Samples for each class' + str(Base.training_classes_samples))

            logger.info('Test metadata --> ' +
                        ' ImageShape:' + str(Base.test_image_shape) +
                        ' Classes:' + str(Base.total_testing_classes) +
                        ' Samples for each class' + str(Base.testing_classes_samples))

    @staticmethod
    def load_classes(root):
        if len(root) == 1:
            class_root_dirs = np.array(sorted([os.path.join(root[0], directory)
                                               for directory in os.listdir(root[0])
                                               if os.path.isdir(os.path.join(root[0], directory))]))
        else:
            class_root_dirs = np.array(sorted([os.path.join(root[0], directory)
                                               for directory in root[1:]
                                               if os.path.isdir(os.path.join(root[0], directory))]))

        classes = np.ndarray(shape=(len(class_root_dirs)), dtype=object)
        labels = np.ndarray(shape=(len(class_root_dirs)), dtype=object)

        for index, path_prefix in enumerate(class_root_dirs):
            temp_arr = np.array([os.path.join(path_prefix, filename)
                                 for filename in os.listdir(path_prefix)
                                 if os.path.isfile(os.path.join(path_prefix, filename))])

            classes[index] = temp_arr
            labels[index] = np.array([index] * temp_arr.size)

        num_of_classes = len(classes)
        classes_samples = [len(class_) for class_ in classes]
        image_shape = ((ndimage.imread(classes[0][0], flatten=True)).astype(float)).shape

        return image_shape, num_of_classes, classes_samples, classes, labels

    @staticmethod
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    @staticmethod
    def merge_datasets(classes, labels, total_parts, miss_parts, valid_parts=0):

        combined_classes = np.array([])
        for class_ in classes:
            combined_classes = np.append(combined_classes, class_)

        combined_labels = np.array([])
        for label in labels:
            combined_labels = np.append(combined_labels, label)

        total_samples = len(combined_classes)

        samples_per_part = total_samples // total_parts
        residue_samples = total_samples - samples_per_part * total_parts

        train_samples = samples_per_part * (total_parts - miss_parts - valid_parts) + residue_samples
        valid_samples = samples_per_part * valid_parts

        shuffled_dataset, shuffled_labels = Base.randomize(combined_classes, combined_labels)

        train_dataset = shuffled_dataset[0:train_samples - 1]
        train_labels = shuffled_labels[0:train_samples - 1]

        if valid_parts:
            valid_dataset = shuffled_dataset[train_samples:train_samples + valid_samples - 1]
            valid_labels = shuffled_labels[train_samples:train_samples + valid_samples - 1]
        else:
            valid_dataset, valid_labels = None, None

        return valid_dataset, valid_labels, train_dataset, train_labels

    def get_valid_and_test_batch_size(self, image_shape, train_batch_size, valid_size, test_size, memory_available):
        image_bytes = image_shape[0] * image_shape[1] * 4
        train_batch_bytes = train_batch_size * image_bytes
        remaining_bytes = int(memory_available * self.safety_percentage - train_batch_bytes)

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

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])


class CNN(Base):
    def __init__(self, safety_percentage, train_parts, train_miss, valid_parts, test_parts, test_miss,
                 train_batch_size, patch_size, depth, num_hidden, channels, image_depth, num_steps,
                 learning_rate, data_showing_step,
                 train_root=None,
                 test_root=None):
        Base.__init__(self, safety_percentage, image_depth, train_root, test_root)

        self.valid_batch_size, self.test_batch_size = -1, -1
        self.train_parts = train_parts
        self.train_miss = train_miss
        self.valid_parts = valid_parts
        self.test_parts = test_parts
        self.test_miss = test_miss

        self.train_batch_size = train_batch_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_hidden = num_hidden
        self.channels = channels
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.data_showing_step = data_showing_step

        self.graph = tf.Graph()
        self.tf_train_dataset = -1
        self.tf_train_labels = -1
        self.tf_valid_dataset = -1
        self.tf_test_dataset = -1

        self.logits = -1
        self.loss = -1
        self.optimizer = -1
        self.train_prediction = -1
        self.valid_prediction = -1
        self.test_prediction = -1

    def reformat(self, dataset, labels, image_shape, num_labels, num_channels):
        dataset = dataset.reshape(
            (-1, image_shape[0], image_shape[1], num_channels)).astype(np.float32)
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels

    def load_batch(self, dataset_file_paths, labels, offset, batch_size, image_shape, pixel_depth, num_labels,
                   num_channels):
        batch_data = np.ndarray(shape=((batch_size), image_shape[0], image_shape[1]), dtype=np.float32)

        batch_labels = np.ndarray(shape=(batch_size), dtype=np.int32)

        image_index = 0
        skipped_images = 0
        index = 0
        length_dataset = len(dataset_file_paths)

        last_seen_good_image = np.array([])
        last_seen_good_label = -1

        while image_index < batch_size and (offset + index) < length_dataset:
            try:
                image_data = (ndimage.imread(dataset_file_paths[offset + index], flatten=True).astype(float) -
                              pixel_depth / 2) / pixel_depth

                last_seen_good_image = image_data
                last_seen_good_label = labels[offset + index]

                if image_data.shape != image_shape:

                    if last_seen_good_label != -1:
                        batch_data[image_index, :, :] = last_seen_good_image
                        batch_labels[image_index] = last_seen_good_label
                        image_index += 1

                    skipped_images += 1
                    print('Unexpected image shape: %s' % str(image_data.shape))
                else:
                    batch_data[image_index, :, :] = image_data
                    batch_labels[image_index] = labels[offset + index]
                    image_index += 1
            except IOError as e:
                if last_seen_good_label != -1:
                    batch_data[image_index, :, :] = last_seen_good_image
                    batch_labels[image_index] = last_seen_good_label
                    image_index += 1

                skipped_images += 1
                logger.warn('Skipping unreadable image:' + dataset_file_paths[offset + index] + ': ' + str(e))
            index += 1

        batch_data = batch_data[0:image_index, :, :]
        batch_labels = batch_labels[0:image_index]

        batch_data, batch_labels = self.reformat(batch_data, batch_labels, image_shape, num_labels, num_channels)

        return batch_data, batch_labels, skipped_images

    @staticmethod
    def model(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases,
              layer4_weights, layer4_biases):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    def setup(self):
        logger.info("Merging training/validation data-sets")
        Base.valid_dataset, Base.valid_labels, Base.train_dataset, Base.train_labels = Base.merge_datasets(
            Base.train_classes, Base.train_classes_labels, self.train_parts, self.train_miss, self.valid_parts)
        logger.info("Merging test data-sets")
        _, _, Base.test_dataset, Base.test_labels = Base.merge_datasets(Base.test_classes, Base.test_classes_labels,
                                                                        self.test_parts, self.test_miss)

        logger.info("Calculating validation and test minibatch sizes")
        self.valid_batch_size, self.test_batch_size = self.get_valid_and_test_batch_size(Base.train_image_shape,
                                                                                         self.train_batch_size,
                                                                                         len(Base.valid_labels),
                                                                                         len(Base.test_labels),
                                                                                         max_graph_size)

        logger.info("Validation samples: " + str(len(Base.valid_labels)))
        logger.info("Test samples: " + str(len(Base.test_labels)))

        logger.info("Validation mini-batch size: " + str(self.valid_batch_size))
        logger.info("Test mini-batch size: " + str(self.test_batch_size))

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(
                tf.float32,
                shape=(self.train_batch_size, Base.train_image_shape[0], Base.train_image_shape[1], self.channels))
            self.tf_train_labels = tf.placeholder(tf.float32,
                                                  shape=(self.train_batch_size, Base.total_training_classes))
            self.tf_valid_dataset = tf.placeholder(
                tf.float32,
                shape=(self.valid_batch_size, Base.train_image_shape[0], Base.train_image_shape[1], self.channels))
            self.tf_test_dataset = tf.placeholder(
                tf.float32,
                shape=(self.test_batch_size, Base.test_image_shape[0], Base.test_image_shape[1], self.channels))

            # Variables.
            layer1_weights = tf.Variable(tf.truncated_normal(
                [self.patch_size, self.patch_size, self.channels, self.depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([self.depth]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [self.patch_size, self.patch_size, self.depth, self.depth], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [Base.train_image_shape[0] // 4 * Base.train_image_shape[1] // 4 * self.depth, self.num_hidden],
                stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            layer4_weights = tf.Variable(tf.truncated_normal(
                [self.num_hidden, Base.total_training_classes], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[Base.total_training_classes]))

            # Training computation.
            self.logits = CNN.model(self.tf_train_dataset, layer1_weights, layer1_biases, layer2_weights, layer2_biases,
                                    layer3_weights, layer3_biases, layer4_weights, layer4_biases)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(CNN.model(self.tf_valid_dataset, layer1_weights, layer1_biases,
                                                            layer2_weights, layer2_biases,
                                                            layer3_weights, layer3_biases, layer4_weights,
                                                            layer4_biases))
            self.test_prediction = tf.nn.softmax(CNN.model(self.tf_test_dataset, layer1_weights, layer1_biases,
                                                           layer2_weights, layer2_biases,
                                                           layer3_weights, layer3_biases, layer4_weights,
                                                           layer4_biases))

    def run(self):
        self.setup()

        skipped_images = 0

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            logger.info('Session Initialized')
            for step in range(self.num_steps):
                offset = (step * self.train_batch_size) % (self.train_labels.shape[0] - self.train_batch_size)
                logger.info('Loading Batch for step ' + str(step))
                batch_data, batch_labels, skipped = self.load_batch(Base.train_dataset, Base.train_labels,
                                                                    offset + skipped_images,
                                                                    self.train_batch_size,
                                                                    (Base.train_image_shape[0],
                                                                     Base.train_image_shape[1]),
                                                                    Base.image_depth,
                                                                    Base.total_training_classes, self.channels)
                skipped_images += skipped
                feed_dict = {self.tf_train_dataset: batch_data, self.tf_train_labels: batch_labels}
                _, l, predictions = session.run(
                    [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % self.data_showing_step == 0):
                    # print('Minibatch loss at step %d: %f' % (step, l))
                    # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    miniBatchAccuracy = Base.accuracy(predictions, batch_labels)

                    # ---- Calculating valid accuracy
                    valid_steps = len(Base.valid_labels) // self.valid_batch_size
                    valid_skipped_images = 0
                    total_valid_checked = 0
                    valid_correct_sum = 0
                    for valid_batch_step in range(valid_steps):
                        valid_offset = (valid_batch_step * self.valid_batch_size)
                        valid_batch_data, valid_batch_labels, valid_skipped = self.load_batch(Base.valid_dataset,
                                                                                              Base.valid_labels,
                                                                                              valid_offset + valid_skipped_images,
                                                                                              self.valid_batch_size,
                                                                                              (
                                                                                              Base.train_image_shape[0],
                                                                                              Base.train_image_shape[
                                                                                                  1]),
                                                                                              Base.image_depth,
                                                                                              Base.total_training_classes,
                                                                                              self.channels)

                        valid_skipped_images += valid_skipped

                        total_valid_checked += len(valid_batch_labels)
                        valid_feed_dict = {self.tf_valid_dataset: valid_batch_data}
                        valid_correct_sum += np.sum(
                            np.argmax(session.run(self.valid_prediction, feed_dict=valid_feed_dict), 1)
                            == np.argmax(valid_batch_labels, 1))
                        validationAccuracy = (100.0 * valid_correct_sum / total_valid_checked)

                    # print('Validation accuracy: %.1f%%' % (100.0 * valid_correct_sum / total_valid_checked))
                    # print('Step %03d  Acc Minibatch: %03.2f%% \t Acc Val: %03.2f%% \t Minibatch loss %f' % (step, miniBatchAccuracy, validationAccuracy, l))
                    logger.info('Step %03d  Acc Minibatch: %03.2f%%  Acc Val: %03.2f%%  Minibatch loss %f' % (
                        step, miniBatchAccuracy, validationAccuracy, l))

            # ---- Calculating test accuracy
            test_steps = len(Base.test_labels) // self.test_batch_size
            test_skipped_images = 0
            total_test_checked = 0
            test_correct_sum = 0
            for test_batch_step in range(test_steps):
                test_offset = (test_batch_step * self.test_batch_size)
                test_batch_data, test_batch_labels, test_skipped = self.load_batch(Base.test_dataset, Base.test_labels,
                                                                                   test_offset + test_skipped_images,
                                                                                   self.test_batch_size,
                                                                                   (Base.test_image_shape[0],
                                                                                    Base.test_image_shape[1]),
                                                                                   Base.image_depth,
                                                                                   Base.total_testing_classes,
                                                                                   self.channels)

                test_skipped_images += test_skipped

                total_test_checked += len(test_batch_labels)
                test_feed_dict = {self.tf_test_dataset: test_batch_data}
                test_correct_sum += np.sum(np.argmax(session.run(self.test_prediction, feed_dict=test_feed_dict), 1)
                                           == np.argmax(test_batch_labels, 1))

            # print('Test accuracy: %.1f%%' % (100.0 * test_correct_sum / total_test_checked))
            logger.info('Test accuracy: %.1f%%' % (100.0 * test_correct_sum / total_test_checked))

train = ['/home/shams/Desktop/notMNIST_large', 'A', 'B']
test = ['/home/shams/Desktop/notMNIST_small', 'A', 'B']

safety_percentage = 0.8

train_parts = 200
train_miss = 10
valid_parts = 5
test_parts = 110
test_miss = 100

batch_size = 32
patch_size = 5
depth = 16
num_hidden = 64
channels = 1
image_depth = 255
num_steps = 101
learning_rate = 0.05
data_showing_step = 2

cnn = CNN(safety_percentage, train_parts, train_miss, valid_parts, test_parts, test_miss, batch_size, patch_size,
          depth, num_hidden, channels, image_depth, num_steps, learning_rate, data_showing_step, train, test)
cnn.run()
