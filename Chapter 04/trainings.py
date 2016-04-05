from models import *

#train = ['/home/shams/Desktop/R25_MINI/R25_MINI/TRAIN']
#test = ['/home/shams/Desktop/R25_MINI/R25_MINI/TEST']

train = ['/home/shams/Desktop/notMNIST_large']
test = ['/home/shams/Desktop/notMNIST_small']

safety_percentage = 0.8

train_parts = 200
train_miss = 10
valid_parts = 5
test_parts = 110
test_miss = 10

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
channels = 1
image_depth = 255
num_steps = 3001
learning_rate = 0.05
data_showing_step = 500

cnn = CNN(safety_percentage, train_parts, train_miss, valid_parts, test_parts, test_miss, batch_size, patch_size,
          depth, num_hidden, channels, image_depth, num_steps, learning_rate, data_showing_step, train, test)
cnn.run()