from imutils import paths
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to directory of images")
ap.add_argument("-a", "--annot", required=True,
                help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of the charator
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i+1,len(imagePaths)))

    try:
        #load image and convert it to grayscale, then pad the image
        #to ensure digits caught on the border of the image
        #are retained
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8,8, cv2.BORDER_REPLICATE)
        #threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

        # find contour in the image, keeping only the four largest ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea(), reverse=True)[:4] #sort contour by area
        # keep 4 largest one

        # given our contours we can extract each of them by computing the bounding box
        # loop over the contours
        for c in cnts:
            # compute bouding box for the contour then extract the digit
            (x,y, w,h) = cv2.boundingRect(c)
            roi = gray[y-5: y+h, x-5: x+w+5]

            #display the character, make it larger enough for us to see
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            if key == ord("q"):
                print("[INFO ignoring character")
                continue

            key = chr(key).upper()
            dirpath = os.path.sep.join([args["annot"], key])

            #if the out put directory does not exist, create it
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

            # write labeled character to file
            count = counts.get(key,1)
            p = os.path.sep.join([dirpath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            #increment the count for current key
            counts[key] = count+1

    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    except:
        print("[INFO] skipping image")

            



