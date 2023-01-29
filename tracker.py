from operator import itemgetter

import cv2
import numpy as np
cap = cv2.VideoCapture("result_moneta.mp4")
draw = cv2.VideoCapture("assets\moneta.mp4")


while True:
    topLCrn = [None, None]
    botRCrn = [None, None]
    ret, frame = cap.read()
    x, output = draw.read()

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
            output = cv2.rectangle(output, (maxY, maxX), (minY, minX), (255, 0, 0), 3)

        cv2.imshow('frame', output)
        if cv2.waitKey(27) == 27:  # 27 - esc
            break
    else:
        break

cap.release()
draw.release()
cv2.destroyAllWindows()