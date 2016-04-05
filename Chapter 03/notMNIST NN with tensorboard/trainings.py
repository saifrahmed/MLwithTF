from models import *

#train = ['/home/shams/Desktop/R25_MINI/R25_MINI/TRAIN']
#test = ['/home/shams/Desktop/R25_MINI/R25_MINI/TEST']

train = ['/home/shams/Desktop/notMNIST_large']
test = ['/home/shams/Desktop/notMNIST_small']

safety_percentage = 0.8

train_parts = 200
train_miss = 10
valid_parts = 3
test_parts = 110
test_miss = 50

batch_size = 256
hidden_nodes = 1024
image_depth = 255
num_steps = 15001
learning_rate = 0.1
data_showing_step = 500

nn = NN(safety_percentage, train_parts, train_miss, valid_parts, test_parts, test_miss, batch_size, hidden_nodes,
          image_depth, num_steps, learning_rate, data_showing_step, train, test)
nn.run()