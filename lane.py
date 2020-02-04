import cv2
import numpy as np
from matplotlib import pyplot as plt


def do_canny(frame, kernel_size):
    # Turn image gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Do GaussianBlur
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Canny, get the edges
    # edges = cv2.Canny(blur_gray, 50, 100)

    edges = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    return edges


def do_polygon(frame, width, height):

    # A polygon area contain main lane information
    polygons = np.array([
        [(0, height),
         (width, height),
         (width, height - 150),
         (width//2, 200),
         (0, height-150)]
    ])

    # A matrix filled with 0, same size with frame
    mask = np.zeros_like(frame)

    # Area of polygon filled with 1s
    cv2.fillPoly(mask, polygons, 255)

    # get 1 part from frame
    polygon_area = cv2.bitwise_and(frame, mask)

    return polygon_area


if __name__ == "__main__":

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
    kernel = 5

    while cap.isOpened():
        # Read video, get (if read success) and (next frame)
        ret, frames = cap.read()
        if not ret:
            break

        canny = do_canny(frames, kernel)
        polygon = do_polygon(canny, size[0], size[1])
        cv2.imshow("polygon", canny)

        # Output frames
        # cv2.imwrite('./images/frame' + str(i) + '.jpg', canny)
        # Output video
        # out.write(canny)
        # Frame number counter + 1
        i += 1
        cv2.waitKey(10)

    cap.release()
    out.release()