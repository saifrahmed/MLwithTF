import sys, os

print(os.path.realpath('..'))
print(__file__)
sys.path.append(os.path.realpath('..'))

from book_code.data_utils import *

not_mnist, image_size, num_of_classes = prepare_not_mnist_dataset()