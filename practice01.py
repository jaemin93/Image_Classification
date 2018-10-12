import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
image_path = 'C:/Users/iceba/develop/deeplearning/code/image_classification/animals/cat/cat (1).jpg'
img = cv2.imread(image_path)
plt.imshow(img)
plt.show()
b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
cv2.imwrite('blue.jpg', b)
cv2.imwrite('green.jpg', g)
cv2.imwrite('red.jpg', r)
img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge

plt.imshow(img2)
plt.show()
def plot_images(image):
    #Create figure with 4x4 sub-plots.
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        row = i // 4
        col = i % 4
        image_frag = image[row*50:(row + 1)*50, col*50:(col+1)*50, :]
        ax.imshow(image_frag)
        #name = 'convert' + str(i) + '.jpg'
        #cv2.imwrite(name, image_frag)

        xlabel = '{},{}'.format(row, col)
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

plot_images(img2)