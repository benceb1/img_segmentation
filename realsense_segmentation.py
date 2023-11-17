import cv2 as cv
import numpy as np
import time
import random as rng
import pyrealsense2 as rs
from cv_bridge import CvBridge
import matplotlib as plt

sleep_time = 1/30
contourThreshold = 50
color = (66, 236, 245)

# Init RealSense
width = 640
height = 480
fps = 30
exposure = 600.0
clipping_distance_in_meters = 0.30

context = rs.context()
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

bridge = CvBridge()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def get_segmented_image(img, aligned_depth_frame):
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
        center_x = int(sum(x_coordinates) / len(h_contour))
        center_y = int(sum(y_coordinates) / len(h_contour))

        d = aligned_depth_frame.get_distance(center_x, center_y)
        print(d)
        
        cv.circle(drawing, (center_x, center_y), 5, (0,0,255), -1)


    return drawing

def process_stream():
    # Start streaming
    profile = pipeline.start(config)

    # Set the camera exposure manually.
    rgb_cam_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    rgb_cam_sensor.set_option(rs.option.exposure, exposure)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # # We will be removing the background of objects more than
    # #  clipping_distance_in_meters meters away
    # clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
         while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            #depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            

            cv.imshow('frame', color_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
            pipeline.stop()


process_stream()