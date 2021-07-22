from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, EfficientNetB7
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import os

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

print("Loading images...")
imagePaths = list(paths.list_images("/home/quan/PycharmProjects/DL_Python/computer_vision2/data/jpg"))
labels = []
j = 0
#Each flower category has 80 images
for (i, p) in enumerate(imagePaths):

    if i%80==0: j += 1
    labels.append(j)




aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

#load data from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
(data, _) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#convert labels from integer to vectors
trainY = LabelEncoder().fit_transform(trainY)
testY = LabelEncoder().fit_transform(testY)

#load the EfficientNet, ensuring the head FClayer sets are left off
baseModel = VGG16(weights='imagenet', include_top=False,
                           input_tensor=Input(shape=(224,224,3)))

# initialize new head of network, followed by a softmax classifier
classNames = [x for x in np.unique(labels)]
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

#place head FCmodel on top of base model
model = Model(inputs=baseModel.input, outputs=headModel)

# #freeze the base model and just train the FClayer
# for layer in baseModel.layers:
#     layer.trainable = False
#
# opt = Adam(lr=0.01)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
#               metrics=['accuracy'])
#
# model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
#                     validation_data=(testX, testY), epochs=25,
#                     steps_per_epoch=len(trainX)//32, verbose=1)
#
# print("Evaluating after initialization...")
# predictions = model.predict(testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1),
#         predictions.argmax(axis=1), target_names=classNames))


#unfreeze base model and train both of them
for layer in baseModel.layers:
    layer.trainable = True

print("Re-compiling model...")
opt = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])


model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=25,
                    steps_per_epoch=len(trainX)//32, verbose=1)

print("Evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classNames))

print("Serializing model...")
model.save('/home/quan/PycharmProjects/DL_Python/computer_vision2/data')