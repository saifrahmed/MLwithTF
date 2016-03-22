from __future__ import print_function
import matplotlib.pyplot as plt
import os
import sys
import tarfile
import numpy as np
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import logging

def load_classes(root):
    class_root_dirs = np.array(sorted([os.path.join(root, directory) 
                               for directory in os.listdir(root) 
                               if os.path.isdir(os.path.join(root, directory))]))
    
    classes = np.ndarray(shape=(len(class_root_dirs),), dtype=object)
    labels = np.ndarray(shape=(len(class_root_dirs),), dtype=object)

    for index, path_prefix in enumerate(class_root_dirs):
        temp_arr = np.array([os.path.join(path_prefix, filename)
                             for filename in os.listdir(path_prefix) 
                             if os.path.isfile(os.path.join(path_prefix, filename))])

        classes[index] = temp_arr  
        labels[index] = np.array([index]*temp_arr.size)
    
    return classes, labels

def make_arrays(nb_rows):
  if nb_rows:
    dataset = np.ndarray(nb_rows, dtype=object)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(classes, train_size, valid_size=0):
  num_classes = len(classes)
  valid_dataset, valid_labels = make_arrays(valid_size)
  train_dataset, train_labels = make_arrays(train_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, class_ in enumerate(classes):
    if valid_dataset is not None:
        valid_letter = class_[:vsize_per_class]
        valid_dataset[start_v:end_v] = valid_letter
        valid_labels[start_v:end_v] = label
        start_v += vsize_per_class
        end_v += vsize_per_class
                    
        train_letter = class_[vsize_per_class:end_l]
        train_dataset[start_t:end_t] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    
  return valid_dataset, valid_labels, train_dataset, train_labels

np.random.seed(133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def pickle_data(filename, 
                train_dataset, train_labels, 
                valid_dataset, valid_labels, 
                test_dataset, test_labels):
    try:
      f = open(filename, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', filename, ':', e)
      raise





train_classes, train_labels = load_classes('notMNIST_large')
test_classes, test_labels = load_classes('notMNIST_small')

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_classes, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_classes, test_size)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

pickle_data('notMNIST.pickle',
            train_dataset, train_labels, 
            valid_dataset, valid_labels, 
            test_dataset, test_labels)
