from tensorflow.keras.applications import VGG16, EfficientNetB4
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from tensorflow.keras.utils import to_categorical
import random
import tqdm
import h5py
import numpy as np
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey='images', bufsize=1024):

        if os.path.exists(outputPath):
            os.remove(outputPath)


        # open hdf5 database for writing and create 2 dataset:
        #   one to store images/features and another to store class labels
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],),
                                             dtype='int')
        # store the buffer size then initialize buffer itself along with the index
        # into the dataset
        self.bufsize = bufsize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        #add rows and label to buffer
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        #check if buffer need to be flush to disk
        if len(self.buffer['data']) >= self.bufsize:
            self.flush()

    def flush(self):
        # write the buffer to disk then reset buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, classLabels):
        #create a dataset to store the actual class label names
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset('label_names', (len(classLabels),),dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        #check to see if there are any other entries in the buffer that need to be flushed to disk
        if len(self.buffer['data']) > 0:
            self.flush()
        # close dataset
        self.db.close()





bs = 32 #batch size
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(('/home/quan/PycharmProjects/DL_Python/data/animals'))) # datasetpath
random.shuffle(imagePaths)

# extract the class labels from image paths then encode label
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)


# load network
print("[INFO] loading network...")
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print(model.summary())

#initialize HDF5 dataset writer, then store class label names in dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), '/home/quan/PycharmProjects/DL_Python/computer_vision2/data/animals.hdf5', dataKey='features')
dataset.storeClassLabels(le.classes_)

for i in np.arange(0, len(imagePaths), bs):
    # extract batch of images and label then initialize list of actual image
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    for (j, imagePath) in tqdm.tqdm(enumerate(batchPaths)):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        if isinstance(image, np.ndarray):
            print("true")
        else :print("false")
        image = preprocess_input(image)
        batchImages.append(image)

        # pass image through network and use outputs as actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    # add feature and label to HDF5 dataset
    dataset.add(features, batchLabels)
    print(features.shape, batchLabels.shape)


dataset.close()
#
#
#
