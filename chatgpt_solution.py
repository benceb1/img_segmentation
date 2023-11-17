import cv2
import numpy as np
import time
import pyrealsense2 as rs

sleep_time = 1 / 30
contour_threshold = 50
color = (66, 236, 245)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

def get_segmented_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_contours = []
    drawing = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        if len(contours[i]) > contour_threshold:
            highlighted_contours.append(contours[i])
            cv2.drawContours(drawing, contours, i, color, 5, cv2.LINE_8, hierarchy, 0)

    for i, h_contour in enumerate(highlighted_contours):
        # fix the shape of the contour array
        h_contour = h_contour.reshape(h_contour.shape[0], h_contour.shape[2])

        x_coordinates, y_coordinates = zip(*h_contour)
        center_x = sum(x_coordinates) / len(h_contour)
        center_y = sum(y_coordinates) / len(h_contour)

        cv2.circle(drawing, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

    return drawing

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if color_frame:
        # Convert the frame to a numpy array
        frame = np.asanyarray(color_frame.get_data())
        frame = get_segmented_image(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(sleep_time)

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
