import numpy as np
import cv2


class LaneLineDetector:
  # Returns a 2x4 numpy array
  def getLines(self, frame):
    color = self._colorDetection(frame)
    canny = self._doCanny(color, kernel_size=5)
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
    # Change image color mode from RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of yellow and white color in HSV
    lower_yellow = np.array([15, 40, 100])
    upper_yellow = np.array([34, 255, 255])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([179, 30, 255])

    # Two masks for yellow and white
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image, get the yellow only image and white only image
    white_res = cv2.bitwise_and(frame, frame, mask=mask_white)
    yellow_res = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    # Add two image
    final_res = cv2.add(white_res, yellow_res)
    return final_res

  # Transform images for get the curve detetion work
  def _tranform(self, frame, width, height):
    # Two array represent a trapezoid and a rectangle
    # 4 elements are: top-left point, top-right point, button-right point, button-left point
    # We need the trapezoid look like the rectangle after transform
    # The trapezoid have two 45 degree angle at button, calculation here for work for different size of video
    # 0.096 is sqrt(3)/18, which is height of trapezoid divided by width of video
    src = np.float32([(width*2//6, height*3//4 - int(0.096*width)), (width*4//6, height*3//4 - int(0.096*width)), (width*5//6, height*3//4), (width*1//6, height*3//4)])
    dst = np.float32([(width*2//6, height*3//4 - int(0.096*width)), (width*4//6, height*3//4 - int(0.096*width)), (width*4//6, height*3//4), (width*2//6, height*3//4)])

    # Get the transform matrix
    m = cv2.getPerspectiveTransform(src, dst)
    # Transform
    transformed = cv2.warpPerspective(frame, m, (width, height), flags=cv2.INTER_LINEAR)

    return transformed

  # 5 way to process the frame and get information
  def _abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    else:
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

  def _mag_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

  def _dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

  def _hls_select(self, img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
      channel = hls[:,:,0]
    elif channel=='l':
      channel=hls[:,:,1]
    else:
      channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

  def _luv_select(self, img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

  def _thresholding(self, img):
    #setting all sorts of thresholds
    x_thresh = self._abs_sobel_thresh(img, orient='x', thresh_min=70 ,thresh_max=255)
    mag_thresh = self._mag_threshold(img, sobel_kernel=3, mag_thresh=(30, 170))
    dir_thresh = self._dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.5))
    hls_thresh = self._hls_select(img, thresh=(20, 255))
    luv_thresh = self._luv_select(img, thresh=(200, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (luv_thresh == 1)] = 1

    return threshholded