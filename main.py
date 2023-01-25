from operator import itemgetter

import cv2
import numpy as np

cap = cv2.VideoCapture("assets\pilka1.mp4")
while True:
    topLCrn = [None, None]
    botRCrn = [None, None]
    _, frame = cap.read()

    # GAUSSIAN BLUR

    # Gaussian blur kernel
    kernel = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

    # Adding gaussian blur kernel
    blurred = cv2.filter2D(frame, -1, kernel)

    #  sobel filters
    sobelVert = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelHor = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #  applying vertical and horizontal sobel filters
    frameSobelVert = cv2.filter2D(blurred, -1, sobelVert)
    frameSobelHor = cv2.filter2D(blurred, -1, sobelHor)

    #  merging the filters
    sobel = frameSobelVert + frameSobelHor

    # cv2.imshow("Frame", sobel)

    #  Edge direction
    theta = np.arctan2(frameSobelVert, frameSobelHor)
    #  convert to degrees in range -180 - 180
    theta = np.rad2deg(theta)
    #  convert range to 0 - 180
    theta[theta < 0] += 180

    #  NON-MAXIMUM SUPPRESSION - too slow

    #  image size
    w, h = sobel.shape[0], sobel.shape[1]

    #  black image as a starting point
    compressed = np.zeros(sobel.shape)

    # seeking the most intense pixel of given direction
    for x in range(1, w - 1):
        for y in range(1, h - 1):

            #  direction of pixel
            direction = theta[x, y]

            # comparing neighbouring pixels +- 22.5 deg (360 / 16)
            # 0 deg (0 - 22.5 or 157.5 - 180)
            if (0 <= direction.any() < 22.5) or (157.5 <= direction.any() <= 180):
                neighbor1 = sobel[x, y + 1]
                neighbor2= sobel[x, y - 1]
            # 45 deg (22.5 - 67.5)
            elif 22.5 <= direction < 67.5:
                neighbor1 = sobel[x + 1, y - 1]
                neighbor2= sobel[x - 1, y + 1]
            # 90 deg (67.5 - 112.5)
            elif 67.5 <= direction < 112.5:
                neighbor1 = sobel[x + 1, y]
                neighbor2= sobel[x - 1, y]
            # 135 deg (112.5 - 157.5)
            elif 112.5 <= direction < 157.5:
                neighbor1 = sobel[x - 1, y - 1]
                neighbor2= sobel[x + 1, y + 1]

            if (sobel[x, y].all() >= neighbor1.any()) and (sobel[x, y].any() >= neighbor2.any()):
                compressed[x, y] = sobel[x, y]

    cv2.imshow("Frame", compressed)


    #  workaround?
    # limits of edges
    lower = np.array([80, 80, 80])
    upper = np.array([255, 255, 255])

    # Mask image
    mask = cv2.inRange(sobel, lower, upper)

    # Mark selected color range
    sobel[mask > 0] = (0, 0, 255)


    # cv2.imshow("Frame", sobel)

    red = [0, 0, 255]

    # Get X and Y coordinates of all red pixels
    Y, X = np.where(np.all(sobel == red, axis=2))

    coordinates = zip(Y, X)
    c = list(coordinates)

    maxX, maxY, minX, minY = 0, 0, sobel.shape[0], sobel.shape[1]  # Declare coordinates of rectangle corners

    if len(c) > 0:
        maxX = max(c, key=itemgetter(0))[0]
        maxY = max(c, key=itemgetter(1))[1]

        minX = min(c, key=itemgetter(0))[0]
        minY = min(c, key=itemgetter(1))[1]
        frame = cv2.rectangle(frame, (maxY, maxX), (minY, minX), (255, 0, 0), 3)

    # cv2.imshow("Frame", frame)

    if cv2.waitKey(27) == 27:  # 27 - esc
        break

cap.release()
cv2.destroyAllWindows()
