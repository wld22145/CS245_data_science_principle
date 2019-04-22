import time
import cv2
import matplotlib.pyplot as plt

# NOTICE: opencv version must be correct
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

imagename = "antelope_10001.jpg"

def test_descriptors(imagename):
    start = time.time()
    image = cv2.imread(imagename)
    print(imagename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    result = cv2.drawKeypoints(gray_image, kp, None)
    print("key point")
    print(kp)
    print("descriptor")
    print(des)
    print("details of key point")
    for i in range(10):
        print("Key Point No.",i)
        print(kp[i].pt)
        print(kp[i].size)
        print(kp[i].angle)
        print(kp[i].response)
    end = time.time()
    elapsed_time =end - start
    print("Elapsed time getting descriptors {0}".format(elapsed_time))
    print("Number of descriptors found {0}".format(len(des)))
    if des is not None and len(des) > 0:
        print("Dimension of descriptors {0}".format(len(des[0])))
    plt.imshow(result)
    plt.savefig("sift_demo.jpg")
    plt.show()

test_descriptors(imagename)