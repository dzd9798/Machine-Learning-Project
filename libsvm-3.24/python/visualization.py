import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io
from skimage import data, exposure




image = io.imread('/home/dzd/dzd/labwork/face/yaleBExtData/yaleB01/yaleB01_P00A-005E-10.pgm')
normalised_blocks, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
