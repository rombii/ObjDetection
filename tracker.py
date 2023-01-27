from operator import itemgetter

import cv2
import numpy as np
cap = cv2.VideoCapture("result.mp4")


while True:
    topLCrn = [None, None]
    botRCrn = [None, None]
    ret, frame = cap.read()

    if ret == True:
        indices = np.where(frame != [0])  # Get all coordinates where white colour exists

        coordinates = zip(indices[0], indices[1])

        c = list(coordinates)

        maxX, maxY, minX, minY = 0, 0, frame.shape[0], frame.shape[1]  # Declare coordinates of rectangle corners

        if len(c) > 0:
            maxX = max(c, key=itemgetter(0))[0]
            maxY = max(c, key=itemgetter(1))[1]

            minX = min(c, key=itemgetter(0))[0]
            minY = min(c, key=itemgetter(1))[1]
            frame = cv2.rectangle(frame, (maxY, maxX), (minY, minX), (255, 0, 0), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(27) == 27:  # 27 - esc
            break
    else:
        break