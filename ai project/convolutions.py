import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open(r"C:\Users\omerg\Downloads\conv test img.png").convert("L"))

kernel = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
out = []

for y in range(1, img.shape[0]-1):
    col = []
    for x in range(1, img.shape[1]-1):
        res = img[y-1:y+2].T[x-1:x+2].T
        col.append(np.sum(res*kernel))
    out.append(col)

out = np.array(out)
plt.imshow(out, cmap="gray")
plt.show()