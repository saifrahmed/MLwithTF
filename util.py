from __future__ import print_function
import os, sys, tarfile
from six.moves.urllib.request import urlretrieve
import datetime
from os.path import basename


import numpy as np
from IPython.display import display, Image
from scipy import ndimage

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


#trn_set = downloadFile('http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz', 247336696)
#tst_set = downloadFile('http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz', 8458043)
#print ('Test set stored in: ' + tst_set)
#tst_files = maybe_extract(tst_set)
tst_files = "2016.03.03_11.15.54/notMNIST_small"
print ('Test file set stored in: ' + tst_files)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
classFolders = [os.path.join(tst_files, d) for d in os.listdir(tst_files) if os.path.isdir(os.path.join(tst_files, d))]
print (classFolders)
for cf in classFolders:
	print ("\n\nExaming class folder " + cf)
	dataset=loadClass(cf)
	print (dataset.shape)
