import cv2
import numpy as np
from matplotlib import pyplot as plt

# Open video
cap = cv2.VideoCapture('./video/trimmed-Gen5_RU_2019-10-07_07-56-42-0001_m0.avi-.avi')
# Frame number counter
i = 0

# Information of the video
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Video writer
out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size, 0)
# Kernel size for GaussianBlur
kernel_size = 5

while cap.isOpened():
    # Read video, get (if read success) and (next frame)
    ret, frame = cap.read()
    if not ret:
        break
    # Turn image gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Do GaussianBlur
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    # Canny, get the edges
    edges = cv2.Canny(blur_gray,50,100)

    # Output frames
    cv2.imwrite('./images/frame' + str(i) + '.jpg', blur_gray)
    # Output video
    out.write(edges)
    # Frame number counter + 1
    i += 1


cap.release()
out.release()