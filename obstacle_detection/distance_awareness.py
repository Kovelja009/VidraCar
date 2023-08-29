import cv2
import numpy as np


# Alert the driver if the distance to the object is too small using the disparity from the OAK-D camera
# distance_threshold -> white is the closest, black is the furthest
# size_threshold -> percentage of "close" pixels in the ROI
class DistanceAlerter:
    def __init__(self, width=640, height=400, roi_width=320, roi_height=300, distance_threshold=190,
                 size_treshold=0.2):
        self._width = width
        self._height = height
        self._roi_width = roi_width
        self._roi_height = roi_height
        self._distance_threshold = distance_threshold
        self._size_threshold = size_treshold

        roi_coordinates = self._get_roi_coordinates()
        self._upper_left = roi_coordinates[0]
        self._upper_right = roi_coordinates[1]
        self._lower_left = roi_coordinates[2]
        self._lower_right = roi_coordinates[3]

        self._color = 255
        self._thickness = 3

    def _get_roi_coordinates(self):
        center_x = int(self._width / 2)
        center_y = int(self._height / 2)

        calc_width = int(self._roi_width / 2)
        calc_height = int(self._roi_height / 2)

        upper_left = (center_x - calc_width, center_y - calc_height)
        upper_right = (center_x + calc_width, center_y - calc_height)
        lower_left = (center_x - calc_width, center_y + calc_height)
        lower_right = (center_x + calc_width, center_y + calc_height)

        return upper_left, upper_right, lower_left, lower_right

    # in: disparity frame
    # out: distance_frame, should_stop
    def check_distance(self, frame):
        roi = frame[self._upper_left[1]:self._lower_right[1], self._upper_left[0]:self._lower_right[0]]

        # Create a binary mask where pixels greater than the threshold are set to 1, others to 0
        close_mask = frame > self._distance_threshold

        # Count the number of white pixels
        close_pixel_count = np.sum(close_mask)

        is_close = False

        if close_pixel_count / (self._roi_width * self._roi_height) > self._size_threshold:
            is_close = True

        # Draw the rectangle on the image
        cv2.rectangle(frame, self._upper_left, self._lower_right, self._color, self._thickness)

        text_position = (self._upper_left[0] + 10, self._upper_left[1] - 15)
        text = "STOP" if is_close else "GO"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, self._color, 2)

        return frame, is_close
