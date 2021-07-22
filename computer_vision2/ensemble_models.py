import matplotlib
matplotlib.use('Agg')
import scikitplot
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import miniVGG
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow.keras import backend
# from tensorflow._api.v2.compat.v1 import ConfigProto
# from tensorflow._api.v2.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

path_output = '/home/quan/PycharmProjects/DL_Python/output'
path_models = '/home/quan/PycharmProjects/DL_Python/models'

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

# convert labels from integers to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
# initialize label names from cifar-10 Dataset
labelNames = ["airplan", "automobile", 'bird', 'cat', 'deer', 'dog', 'frog',
               "horse", " s h i p " , " t r u c k "]
opt1 = Adam(learning_rate=0.01)
opt2 = SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov=True )
opt3 = RMSprop(learning_rate=0.005)
# initialize the optimizer and model
print("[INFO] training model {}/{}".format(4, 5))
model = miniVGG(width=32, height=32, depth=3, classes=10)
model = model.build()
model.compile(loss='categorical_crossentropy', optimizer=opt1,
              metrics=['accuracy'])

H = model.fit(aug.flow(trainX, trainY, batch_size=64),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64, epochs=40, verbose=2)
# save model to disk
p = [path_models, " model_{}.model".format(4)]
model.save(os.path.sep.join(p))

predictions = model.predict(testX, batch_size=32)
report = classification_report(testY.argmax(axis=1),
                               predictions.argmax(axis=1),
                               target_names=labelNames)
# save classification report to file
p = [path_output, 'model_{}.txt'.format(4)]
f = open(os.path.sep.join(p), 'w')
f.write(report)
f.close()

p = [path_output, 'model_{}.png'.format(4)]
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'],
         label='train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'],
         label='val_loss')
plt.plot(np.arange(0, 40), H.history['accuracy'],
         label='train_accuracy')
plt.plot(np.arange(0, 40), H.history['val_accuracy'],
         label='val_accuracy')
plt.title("Training Loss and Accuracy for model {}".format(2))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(os.path.sep.join(p))
plt.close()

scikitplot.metrics.plot_confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1), figsize=(20, 20))

import tensorflow.keras.models
from sklearn.model_selection import StratifiedKFold
batch_size = 32
no_classes = 7
no_epochs = 25
num_folds = 7

accuracy_list = []
loss_list = []

kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
# model = models.load_model('/content/gdrive/MyDrive/fer2013/fer2013_84.model')
# K-fold Cross Validation model evaluation
fold_idx = 1
model = miniVGG(width=48, height=48, depth=1, classes=7)
model = model.build()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
for train_ids, val_ids in kfold.split(faces, emotions):
    print("Bắt đầu train Fold ", fold_idx)

    # Train model
    model.fit(X[train_ids], y[train_ids], validation_data=(X[val_ids], y[val_ids]),
              batch_size=batch_size,
              epochs=no_epochs)

    # Test và in kết quả
    scores = model.evaluate(X[val_ids], y[val_ids], verbose=0)
    print("Đã train xong Fold ", fold_idx)

    # Thêm thông tin accuracy và loss vào list
    accuracy_list.append(scores[1] * 100)
    loss_list.append(scores[0])

    # Sang Fold tiếp theo
    fold_idx = fold_idx + 1

# In kết quả tổng thể
print('* Chi tiết các fold')
for i in range(0, len(accuracy_list)):
    print(f'> Fold {i + 1} - Loss: {loss_list[i]} - Accuracy: {accuracy_list[i]}%')

print('* Đánh giá tổng thể các folds:')
print(f'> Accuracy: {np.mean(accuracy_list)} (Độ lệch +- {np.std(accuracy_list)})')
print(f'> Loss: {np.mean(loss_list)}')

print('* Chi tiết các fold')
for i in range(0, len(accuracy_list)):
    print(f'> Fold {i + 1} - Loss: {loss_list[i]} - Accuracy: {accuracy_list[i]}%')

print('* Đánh giá tổng thể các folds:')
print(f'> Accuracy: {np.mean(accuracy_list)} (Độ lệch +- {np.std(accuracy_list)})')
print(f'> Loss: {np.mean(loss_list)}')



