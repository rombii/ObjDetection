from operator import itemgetter

import cv2
import numpy as np

cap = cv2.VideoCapture("assets\pilka1.mp4")
while True:
    topLCrn = [None, None]
    botRCrn = [None, None]
    _, frame = cap.read()

    kernel = np.ones((5,5), np.float32)/25

    dst = cv2.filter2D(frame, -1, kernel)

    cv2.imshow("filtered", dst)

    i_x = cv2.Sobel(dst, cv2.CV_64F, 1, 0)

    

    i_y = cv2.Sobel(dst, cv2.CV_64F, 0, 1)



    edges = cv2.Canny(frame, 100, 200)  # Canny edge detection try to implement yourself

    indices = np.where(edges != [0])    # Get all coordinates where white colour exists

    coordinates = zip(indices[0], indices[1])

    c = list(coordinates)

    maxX, maxY, minX, minY = 0, 0, edges.shape[0], edges.shape[1]   # Declare coordinates of rectangle corners

    if len(c) > 0:
        maxX = max(c, key=itemgetter(0))[0]
        maxY = max(c, key=itemgetter(1))[1]

        minX = min(c, key=itemgetter(0))[0]
        minY = min(c, key=itemgetter(1))[1]
        frame = cv2.rectangle(frame, (maxY, maxX), (minY, minX), (255, 0, 0), 3)


    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    if cv2.waitKey(27) == 27: # 27 - esc
        break

cap.release()
cv2.destroyAllWindows()
