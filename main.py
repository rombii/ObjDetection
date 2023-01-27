from operator import itemgetter

import cv2
import numpy as np

cap = cv2.VideoCapture("assets\pilka1.mp4")
result = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(640.0), int(480.0)))
number = 0
while True:
    topLCrn = [None, None]
    botRCrn = [None, None]
    ret, frame = cap.read()

    if ret == True:
        number += 1
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
                # 0 deg (0 - 22.5 or 157.5 - 180
                if (0 <= direction[0] < 22.5) or (157.5 <= direction[0] <= 180):
                    neighbor1 = sobel[x, y + 1]
                    neighbor2 = sobel[x, y - 1]
                # 45 deg (22.5 - 67.5)
                elif 22.5 <= direction[0] < 67.5:
                    neighbor1 = sobel[x + 1, y - 1]
                    neighbor2 = sobel[x - 1, y + 1]
                # 90 deg (67.5 - 112.5)
                elif 67.5 <= direction[0] < 112.5:
                    neighbor1 = sobel[x + 1, y]
                    neighbor2 = sobel[x - 1, y]
                # 135 deg (112.5 - 157.5)
                elif 112.5 <= direction[0] < 157.5:
                    neighbor1 = sobel[x - 1, y - 1]
                    neighbor2 = sobel[x + 1, y + 1]
                if (sobel[x, y][0] >= neighbor1[0]) and (sobel[x, y][0] >= neighbor2[0]):
                    compressed[x, y] = sobel[x, y]
                else:
                    compressed[x, y] = 0

        # Double threshold

        highThreshold = 150
        lowThreshold = 50

        afterThreshold = np.zeros(compressed.shape)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(compressed >= highThreshold)[0], np.where(compressed >= highThreshold)[1]

        weak_i, weak_j = np.where((compressed < highThreshold) & (compressed >= lowThreshold))[0], \
                         np.where((compressed < highThreshold) & (compressed >= lowThreshold))[1]

        afterThreshold[strong_i, strong_j] = strong
        afterThreshold[weak_i, weak_j] = weak

        # cv2.imshow('thresh', res)

        # Applying Hysteresis
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if afterThreshold[i, j][0] == weak:
                    if ((afterThreshold[i + 1, j - 1][0] == strong)
                            or (afterThreshold[i + 1, j][0] == strong)
                            or (afterThreshold[i + 1, j + 1][0] == strong)
                            or (afterThreshold[i, j - 1][0] == strong)
                            or (afterThreshold[i, j + 1][0] == strong)
                            or (afterThreshold[i - 1, j - 1][0] == strong)
                            or (afterThreshold[i - 1, j][0] == strong)
                            or (afterThreshold[i - 1, j + 1][0] == strong)):
                        afterThreshold[i, j] = strong
                    else:
                        afterThreshold[i, j] = 0
        result.write(np.int8(afterThreshold))

        if cv2.waitKey(27) == 27:  # 27 - esc
            break
        print("frame: " + str(number))
    else:
        break
result.release()
cap.release()
cv2.destroyAllWindows()
