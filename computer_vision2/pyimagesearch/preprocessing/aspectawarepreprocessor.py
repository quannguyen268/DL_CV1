import  imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        #grab the dimension of image and then initialize the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if width is smaller than the height, then resize along the width (the smaller dimension) and then update
        # the deltas to crop the height to the desired dimension
        if w<h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        #re-grab the width and height
        (h, w) = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

image_or = cv2.imread('/home/quan/PycharmProjects/DL_Python/computer_vision2/data/jpg/image_0001.jpg')
print(image_or.shape)
cv2.imshow('im', image_or)

image = cv2.resize(image_or, (100,200), interpolation=cv2.INTER_AREA)
cv2.imshow('im0', image)
aap = AspectAwarePreprocessor(100,200)
image = aap.preprocess(image_or)
cv2.imshow('im1', image)
cv2.waitKey(0)
