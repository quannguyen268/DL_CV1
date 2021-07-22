from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel):
    # grap dimension of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]


    # allocate memory for the output image, taking care to "pad"
    # the borders of the input image so the input image so the spatial size are not reduced

    pad = (kW -1) // 2

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE )

    output = np.zeros((iH, iW), dtype="float")

    for h in range(pad, iH + pad ):
        for w in range(pad, iW + pad):
            ROI = image[h - pad:h + pad + 1, w - pad:w + pad + 1]


            k = (ROI*kernel).sum()
            output[h - pad, w - pad] = k


    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')
    return output



# construct average blurring kernels used to smooth an image
smallblur = np.ones((7, 7), dtype="float32")*(1.0 /(7 * 7))
largeblur = np.ones((21, 21), dtype="float32")*(1.0 /(21 * 21))

sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")

laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")

sobelX = np.array(([-1, 0, 1],
                   [-2, 0 ,2],
                   [-1, 0, 1]), dtype='int')
sobelY = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]), dtype='int')
emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype='int')

# construct kernel bank to apply
kernel_bank = (("small_blur", smallblur),
               ("large_blur", largeblur),
               ("sharpen", sharpen),
               ("laplacian", laplacian),
               ("sobel_x", sobelX),
               ("sobel_y", sobelY),
               ("emboss", emboss))

image = cv2.imread('/home/quan/PycharmProjects/OpenCV_practical/157737215_250521136748497_390207163245905894_o.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over kernel bank
for (kernel_name, kernel) in kernel_bank:
    print("Applying {} kernel".format(kernel_name))
    convolveOutput = convolve(gray,kernel)
    opencv2Output = cv2.filter2D(gray, -1, kernel)

    #show output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernel_name), convolveOutput)
    cv2.imshow("{} - opencv".format(kernel_name), opencv2Output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


