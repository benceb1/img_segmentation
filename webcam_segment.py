import cv2 as cv
import numpy as np
import time
import random as rng

sleep_time = 1/30
contourThreshold = 50
color = (66, 236, 245)
cap = cv.VideoCapture(0)

def get_segmented_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, dst = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    highlighed_contours = []
    drawing = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        if len(contours[i]) > contourThreshold:
            highlighed_contours.append(contours[i])
            cv.drawContours(drawing, contours, i, color, 5, cv.LINE_8, hierarchy, 0)

    for i, h_contour in enumerate(highlighed_contours):
        # fura lett az a lista, ezért változtatni kell rajta
        h_contour = h_contour.reshape(h_contour.shape[0], h_contour.shape[2])

        x_coordinates, y_coordinates = zip(*h_contour)
        center_x = sum(x_coordinates) / len(h_contour)
        center_y = sum(y_coordinates) / len(h_contour)

        cv.circle(drawing, (int(center_x), int(center_y)), 5, (0,0,255), -1)

    return drawing

while True:
    ret, frame = cap.read()
    
    if (ret):
        frame = get_segmented_image(frame)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(sleep_time)

cap.release()
cv.destroyAllWindows()

