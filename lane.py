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

    edges = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)

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


def calculate_lines(frame, lines):
    # 建立两个空列表，用于存储左右车道边界坐标
    left = []
    right = []

    # 循环遍历lines
    for line in lines:
        # 将线段信息从二维转化能到一维
        x1, y1, x2, y2 = line.reshape(4)

        # 将一个线性多项式拟合到x和y坐标上，并返回一个描述斜率和y轴截距的系数向量
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0] #斜率
        y_intercept = parameters[1] #截距

        # 通过斜率大小，可以判断是左边界还是右边界
        # 很明显左边界slope<0(注意cv坐标系不同的)
        # 右边界slope>0
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    # 将所有左边界和右边界做平均，得到一条直线的斜率和截距
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    # 将这个截距和斜率值转换为x1,y1,x2,y2
    left_line = calculate_coordinate(frame, parameters=left_avg)
    right_line = calculate_coordinate(frame, parameters=right_avg)

    return np.array([left_line,right_line])


# 将截距与斜率转换为cv空间坐标
def calculate_coordinate(frame, parameters):
    # 获取斜率与截距
    slope, y_intercept = parameters

    # 设置初始y坐标为自顶向下(框架底部)的高度
    # 将最终的y坐标设置为框架底部上方150
    y1 = frame.shape[0]
    y2 = int(y1-150)
    # 根据y1=kx1+b,y2=kx2+b求取x1,x2
    x1 = int((y1-y_intercept)/slope)
    x2 = int((y2-y_intercept)/slope)
    return np.array([x1,y1,x2,y2])


if __name__ == "__main__":

    # Open video
    cap = cv2.VideoCapture('./video/Gen5_RU_2019-10-07_07-56-42-0001_m0.avi')
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
        polygon = do_polygon(~canny, size[0], size[1])

        hough = cv2.HoughLinesP(polygon, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        for line in hough:
            x1, y1, x2, y2 = line[0]
            cv2.line(frames, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #img = np.array(hough, dtype=np.uint8)
        cv2.imshow("polygon", frames)

        # 将从hough检测到的多条线平均成一条线表示车道的左边界，
        # 一条线表示车道的右边界
        #lines = calculate_lines(frames, hough)

        # Output frames
        cv2.imwrite('./images/frame' + str(i) + '.jpg', canny)
        # Output video
        #out.write(canny)
        # Frame number counter + 1
        i += 1
        cv2.waitKey(10)

    cap.release()
    #out.release()