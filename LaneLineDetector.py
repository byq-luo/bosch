import numpy as np
import cv2

class LaneLineDetector:
  # Returns a 2x4 numpy array
  def getLines(self, frame):
    canny = self._doCanny(frame, kernel_size=5)
    height, width, depth = frame.shape
    polygon = self._doPolygon(canny, width, height)

    hough = cv2.HoughLinesP(polygon, 1, np.pi/180, 50, minLineLength=70, maxLineGap=10)

    if hough is None:
      return np.zeros((2,4))

    return self._calculateLines(frame, hough)

  # Do canny and gaussian for input frames
  def _doCanny(self, frame, kernel_size):
    # Turn image gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Do GaussianBlur
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Canny, get the edges
    # edges = cv2.Canny(blur_gray, 50, 100)

    edges = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    return edges

  # Remove useless parts, only the parts that may contain lane information
  def _doPolygon(self, frame, width, height):

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
  def _calculateLines(self, frame, lines):
    # Array for left lane and right lane
    left = []
    right = []

    for line in lines:
      x1, y1, x2, y2 = line[0]

      # Get slope and y interception of line
      if x1 == x2:
        slope = float('inf')
        y_intercept = 0.0
      else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = slope * (0-x1) + y1

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
      left_line = self._calculateCoordinate(frame, parameters=left_avg)
    if np.size(right) == 0:
      right_line = np.array([0, 0, 0, 0])
    else:
      right_avg = np.average(right, axis=0)
      right_line = self._calculateCoordinate(frame, parameters=right_avg)

    return np.array([left_line, right_line])

  # From slop and y interception, get (x, y) for two end point
  def _calculateCoordinate(self, frame, parameters):
    slope, y_intercept = parameters

    # Line start from button end with half of the height of frame
    y1 = frame.shape[0]
    y2 = int(y1/2)
    # Get x1, x2 according to y1, y2, slope and intersection
    x1 = int((y1-y_intercept)/slope)
    x2 = int((y2-y_intercept)/slope)
    return [x1, y1, x2, y2]

  # Detect lane colors: white and yellow
  def _colorDetection(self, frame):
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
      #cv2.imshow("images", np.hstack([frame, output]))
