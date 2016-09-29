import sys, os
import tensorflow as tf
sys.path.append(os.path.realpath('../..'))
from book_code.data_utils import *
from book_code.logmanager import *

prepare_dr_dataset(save_space=True)