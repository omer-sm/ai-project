import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def green_red_counter():
    red1, green1 = 0,0
    raccoon = Image.open(r'unit10\raccoon.png')
    plt.imshow(raccoon)
    plt.ion()
    plt.show()
    array = np.array(raccoon)
    tic = time.time()
    # non vectorized code
    for r in range(len(array)):
        for c in range(len(array[0])):
            if (array[r][c][0] > array[r][c][1]):
                red1 += 1
            elif (array[r][c][0] < array[r][c][1]):
                green1 += 1
    toc = time.time()
    print ("Non Vectorized version: red = " + str(red1) + ", green = " + str(green1)+ ". It took " + str(1000*(toc-tic)) + "ms")
    # vectorized code
    tic = time.time()
    redArr = array[:,:,0].flatten('F')
    greenArr = array[:,:,1].flatten('F')
    red1 = np.sum(redArr > greenArr)
    green1 = np.sum(redArr < greenArr)
    toc = time.time()
    print ("Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(1000*(toc-tic)) + "ms")

def image_to_arr(path):
    img = Image.open(repr(path)[1:-1])
    arr = np.array(img)
    flatArr = arr.flatten('F')
    reshapeArr = arr.reshape(-1)
    print("flat array: ", flatArr)
    print("reshape array: ", reshapeArr)

image_to_arr("unit10\raccoon.png")