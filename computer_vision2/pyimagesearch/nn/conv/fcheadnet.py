from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        #initialize the head model that will be placed on top of the base then add a FClayer
        headModel = baseModel.output
        headModel = Flatten(name='flatten')(headModel)
        headModel = Dense(D, activation='swish')(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation='softmax')(headModel)

        return headModel

