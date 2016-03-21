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
import logging

def load_classes(root):
    class_root_dirs = np.array(sorted([os.path.join(root, directory) 
                               for directory in os.listdir(root) 
                               if os.path.isdir(os.path.join(root, directory))]))
    
    classes = np.ndarray(shape=(len(class_root_dirs),), dtype=object)

    for index, path_prefix in enumerate(class_root_dirs):
        temp_arr = np.array([os.path.join(path_prefix, filename)
                             for filename in os.listdir(path_prefix) 
                             if os.path.isfile(os.path.join(path_prefix, filename))])

        classes[index] = temp_arr    
    
    return classes
