from __future__ import print_function
import os, sys, tarfile
from six.moves.urllib.request import urlretrieve
import datetime
from os.path import basename


import numpy as np
from IPython.display import display, Image
from scipy import ndimage

import pickle 

def downloadFile(fileURL, expected_size):
	timeStampedDir=datetime.datetime.now().strftime("%Y.%m.%d_%I.%M.%S")
	os.makedirs(timeStampedDir)
	fileNameLocal = timeStampedDir + "/" + fileURL.split('/')[-1]
	print ('Attempting to download ' + fileURL)
	print ('File will be stored in ' + fileNameLocal)
	filename, _ = urlretrieve(fileURL, fileNameLocal)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_size:
		print('Found and verified', filename)
	else:
		raise Exception('Could not get ' + filename)
	return filename

def extractFile(filename):
	timeStampedDir=datetime.datetime.now().strftime("%Y.%m.%d_%I.%M.%S")
	tar = tarfile.open(filename)
	sys.stdout.flush()
	tar.extractall(timeStampedDir)
	tar.close()
	return timeStampedDir + "/" + os.listdir(timeStampedDir)[0]

def loadClass(folder):
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  return dataset[0:image_index, :, :]

'''    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
'''
def makePickle(imgSrcPath):
	print ("Pickling " + imgSrcPath)
	data_folders = [os.path.join(imgSrcPath, d) for d in os.listdir(imgSrcPath) if os.path.isdir(os.path.join(imgSrcPath, d))]
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		print('Pickling %s.' % set_filename)
		dataset = loadClass(folder)
		try:
			with open(set_filename, 'wb') as f:
				pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', set_filename, ':', e)
	return dataset_names

#trn_set = downloadFile('http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz', 247336696)
#tst_set = downloadFile('http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz', 8458043)
#print ('Test set stored in: ' + tst_set)
#trn_files = extractFile(trn_set)
#tst_files = extractFile(tst_set)
#print ('Test file set stored in: ' + tst_files)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

#print (classFolders)
trn_files="2016.03.04_01.48.34/notMNIST_large"
tst_files="2016.03.04_01.57.18/notMNIST_small"
picklenamesTrn = makePickle(trn_files)
picklenamesTst = makePickle(tst_files)

#for cf in classFolders:
#	print ("\n\nExaming class folder " + cf)
#	dataset=loadClass(cf)
#	print (dataset.shape)






def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  picklenamesTrn, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(picklenamesTst, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
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
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


#############################
