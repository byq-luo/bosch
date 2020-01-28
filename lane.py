import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('1540915995809.jpeg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

cap = cv2.VideoCapture('./video/trimmed-Gen5_RU_2019-10-07_07-56-42-0001_m0.avi-.avi')
i=0
img_array = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    #cv2.imwrite('./images/frame' + str(i) + '.jpg', frame)
    height, width, layer = frame.shape
    edges = cv2.Canny(frame,100,200)
    size = (width, height)
    img_array.append(frame)

    i+=1

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
cap.release()
out.release()