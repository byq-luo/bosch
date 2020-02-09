import cv2
import numpy as np
from matplotlib import pyplot as plt


# Do canny and gaussian for input frames
def do_canny(frame, kernel_size):
    # Turn image gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Do GaussianBlur
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Canny, get the edges
    # edges = cv2.Canny(blur_gray, 50, 100)

    edges = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    return edges


# Remove useless parts, only the parts that may contain lane information
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


# From input line information, calculate the left lane and right lane equation
def calculate_lines(frame,lines):
    # Array for left lane and right lane
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Get slope and y interception of line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]

        # Add good lines to array
        if abs(slope) > 0.5:
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))

    # If no good lines detected, do not output
    if np.size(left) == 0:
        left_line = np.array([0, 0, 0, 0])
    else:
        # get average of good lines
        left_avg = np.average(left, axis=0)
        # Convert value to x and y
        left_line = calculate_coordinate(frame, parameters=left_avg)
    if np.size(right) == 0:
        right_line = np.array([0, 0, 0, 0])
    else:
        right_avg = np.average(right, axis=0)
        right_line = calculate_coordinate(frame, parameters=right_avg)

    return np.array([left_line, right_line])


# From slop and y interception, get (x, y) for two end point
def calculate_coordinate(frame, parameters):
    slope, y_intercept = parameters

    # Line start from button end with half of the height of frame
    y1 = frame.shape[0]
    y2 = int(y1/2)
    # Get x1, x2 according to y1, y2, slope and intersection
    x1 = int((y1-y_intercept)/slope)
    x2 = int((y2-y_intercept)/slope)
    return np.array([x1, y1, x2, y2])


# visualize the lines
def visualize_lines(frame, lines):
    lines_visualized = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lines_visualized, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return lines_visualized


# Detect lane colors: white and yellow
def color_detection(frame):
    # Color boundaries for white and yellow lane color
    boundaries = [([224, 224, 224], [255, 255, 255]), ([0, 204, 204], [100, 255, 255])]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        # show the images
        cv2.imshow("images", np.hstack([frame, output]))



if __name__ == "__main__":

    # Open video
    cap = cv2.VideoCapture('./video/Gen5_RU_2019-10-07_07-56-42-0001_m0.avi')
    #cap = cv2.VideoCapture('./online_vid.mp4')
    # Frame number counter
    i = 0

    # Information of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Video writer
    #out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size, 0)

    # Kernel size for GaussianBlur
    kernel = 5

    while cap.isOpened():
        # Read video, get (if read success) and (next frame)
        ret, frames = cap.read()
        if not ret:
            break

        canny = do_canny(frames, kernel)
        polygon = do_polygon(canny, size[0], size[1])

        hough = cv2.HoughLinesP(polygon, 1, np.pi/180, 50, minLineLength=70, maxLineGap=10)
        #for line in hough:
            #x1, y1, x2, y2 = line[0]
            #cv2.line(frames, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imshow("polygon", frames)

        lines = calculate_lines(frames, hough)
        lines_visualize = visualize_lines(frames, lines)
        output = cv2.addWeighted(frames,0.6,lines_visualize,1,0.1)
        cv2.imshow("output", output)

        # Output frames
        # cv2.imwrite('./images/frame' + str(i) + '.jpg', canny)

        # Output video
        #out.write(canny)

        # Frame number counter + 1
        i += 1
        cv2.waitKey(10)

    cap.release()
    #out.release()