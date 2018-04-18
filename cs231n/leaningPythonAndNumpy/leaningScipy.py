from scipy.misc import imread , imsave , imresize
img = imread('test.png')
print(img.dtype , img.shape)
img_tinted = img * [1 , 0.95 , 0.9]
img_tinted = imresize(img_tinted , (300 , 300))
imsave('test2.png' , img_tinted)

import numpy as np;
import matplotlib.pyplot as plt

x = np.arange(0 , 2 * np.pi , 0.1)
y = np.sin(x)

plt.plot(x , y)
plt.show()

x = np.arange(0 , 3 * np.pi , 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x , y_sin)
plt.plot(x , y_cos)
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title('sine and cosine')
plt.legend(['sine' , "cosine"])
plt.show()

img = imread("test.png")
img_tinted = img * [1 , 0.95 , 0.9]
plt.subplot(1 , 2 , 1)
plt.imshow(img)
plt.subplot(1 , 2 , 2)
plt.imshow(np.uint8(img_tinted))
plt.show()