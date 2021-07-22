import argparse
import requests
import time
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
ap.add_argument("-n", "--num_images", type=int,
                default=500, help="# of images to download")
args = vars(ap.parse_args())
# url contain captcha images
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

#loop over the image to download
for i in range(0, args["num_images"]):
    try:
        #try to grab a new captcha image
        r = requests.get(url, timeout=60)

        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        #update the counterc
        print("Downloaded: {}".format(p))
        total += 1

    except:
        print("Error downloading image...")

    #insert a small sleep to be courteous to the server
    time.sleep(0.1)