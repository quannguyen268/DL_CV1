import matplotlib
matplotlib.use("Agg")

# import the necessary package
from utils import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path t the output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

#convert label from intergers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(
os.getpid())])

jsonPath = os.path.sep.join([args["output"], "{}.json".format(
os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY),
batch_size=64, epochs=100, callbacks=callbacks, verbose=1)