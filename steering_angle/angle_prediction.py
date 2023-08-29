import cv2
import numpy as np
import math

# Description: This file contains the code for calculating the steering angle of the car.

class State:
    # possible states: 0 <- initial polygon, 1 <- one lane detected, 2 <- two lanes detected
    def __init__(self, height, width):
        self.prev_angle = 0
        self.straight_line = int(width / 2), int(height), int(width / 2), int(height / 2)
        self.prev_line = self.straight_line
        self.prev_state = 0
        self.n_frames = 0
        self.initial_shape = np.array(
            [[(int(width * 0.15), height), (int(width * 0.35), int(height * 0.4)),
              (int(width * 0.6), int(height * 0.4)), (int(width * 0.85), height)]])
        self.last_shape = self.initial_shape
        self.danger = False  # occurs when we don't find any lane


class AngleCalculator:
    def __init__(self, height=480, width=720, resize=1.0, draw_lines=False, decay=0.0, show_additional_windows=False):
        self.show_additional_windows = show_additional_windows
        self.resize = resize
        self.draw_lines = draw_lines
        self.height = int(height * resize)
        self.width = int(width * resize)
        self.EPSYLON = 0.000001
        self.decay = decay
        self.state = State(self.height, self.width)

    def get_angle(self, image):
        if self.resize != 1.0:
            image = self._resize(image)
        canny_image = self._canny(image)
        cropped_image = self._region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30,
                                maxLineGap=200)
        left_line, right_line = self._average_slope_intercept(image, lines)
        central_line, angle = self._angle_calculator(image, left_line, right_line)
        if self.draw_lines:
            image = self._display_lines(image, left_line, right_line, central_line)

        return image, angle

    def _angle_calculator(self, image, left_line, right_line):
        # if angle is 90 degrees than k is infinite, so we add EPSYLON to ensure that we aren't deviding by zero
        k_left = 0
        k_right = 0
        n_left = 0
        n_right = 0

        # if we haven't found any line but have searched whole ROI, just go straight
        if self.state.danger and self.state.prev_state == 0 and left_line is None and right_line is None:
            self.state.prev_line = self.state.straight_line
            self.state.prev_angle = round(0.0, 2)
            return self.state.straight_line, self.state.prev_angle

        # if we haven't found any lines (maybe because of noise in image) but want to keep previous angle
        if left_line is None and right_line is None:
            return self.state.prev_line, self.state.prev_angle
        x1, y1, x2, y2 = 0, 0, 0, 0
        ########################
        # ensuring safety via checking line availability: if we find
        # just one line, we will stick to that one
        if left_line is not None:
            x1, y1, x2, y2 = left_line
        else:
            x1, y1, x2, y2 = self.state.straight_line
        k_left = (y2 - y1) / (x2 - x1 + self.EPSYLON)
        n_left = y1 - (k_left * x1)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
        else:
            x1, y1, x2, y2 = self.state.straight_line
        ########################
        k_right = (y2 - y1) / (x2 - x1 + self.EPSYLON)
        n_right = y1 - (k_right * x1)

        # calculating where 2 lines intersect
        intersection_x = (n_right - n_left) / (k_left - k_right + self.EPSYLON)
        intersection_y = k_left * intersection_x + n_left

        central_line = int(self.width / 2), int(self.height), int(intersection_x), int(intersection_y)
        dx = int(intersection_x - self.width / 2)
        dy = int(intersection_y - self.height)
        theta = math.atan2(dy, dx)
        angle = math.degrees(theta) + 90

        self.state.prev_line = central_line
        curr_angle = round(angle, 2)

        # does smoothing for steering by not allowing for great fluctuation between frames
        adjusted_angle = round(self.state.prev_angle * self.decay + (1 - self.decay) * curr_angle, 2)
        self.state.prev_angle = adjusted_angle

        return central_line, adjusted_angle

    def _dynamic_ROI(self):
        if self.state.n_frames >= 24:  # should aproximately be about 1 sec
            self.state.n_frames = 0
            self.state.prev_state = 0
            self.state.danger = False
            return self.state.initial_shape
        return self.state.last_shape

    # y = kx + n  [k = slope, n = intercept]
    def _average_slope_intercept(self, image, lines):
        left_lines = []
        right_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                # first degree polynomial
                res = np.polyfit((x1, x2), (y1, y2), 1)
                k = res[0]
                n = res[1]
                if k < 0:
                    left_lines.append((k, n))
                else:
                    right_lines.append((k, n))  # [slope1, intercept1]
        #   axis = 0 means we are doing average column wise -> [slope2, intercept2]
        left_line = None
        right_line = None
        if len(left_lines) != 0:
            average_left = np.average(left_lines, axis=0)
            left_line = self._make_coordinates(average_left)
        if len(right_lines) != 0:
            average_right = np.average(right_lines, axis=0)
            right_line = self._make_coordinates(average_right)

        edge_case = False
        # setting state and shape
        ##################### no lines detected
        if len(left_lines) == 0 and len(right_lines) == 0:  # no lines detected in this frame
            edge_case = True
            self.state.n_frames += 1
            if self.state.n_frames >= 10 and self.state.prev_state == 0:  # if we haven't find them for more than 5 frames then we change state
                self.state.last_shape = self.state.initial_shape
                self.state.prev_state = 0
                self.state.danger = False
                self.state.n_frames = 0
            else:  # then just use previous shape <- trying to prevent noise
                if self.state.n_frames == 9 and self.state.danger:
                    self.state.prev_state = 0
                self.state.danger = True
        ##################### detected one line
        if (len(left_lines) == 0 and len(right_lines) != 0) or (len(left_lines) != 0 and len(right_lines) == 0):
            edge_case = True
            self.state.prev_state = 1
            self.state.danger = False
            self.state.n_frames += 1
            if len(left_lines) != 0:
                self.state.last_shape = self._new_shape(left_line)
            else:
                self.state.last_shape = self._new_shape(right_line)

        if not edge_case:  # means we found both lines
            self.state.n_frames = 0
            self.state.danger = False
            self.state.prev_state = 2
            self.state.last_shape = self._new_shape(left_line, right_line)

        return left_line, right_line

    def _region_of_interest(self, image):
        polygon = self._dynamic_ROI()
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)

        if self.show_additional_windows:
            cv2.imshow("mask", mask)
        mask = cv2.bitwise_and(mask, image)

        if self.show_additional_windows:
            cv2.imshow("ROI", mask)

        return mask

    def _canny(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # blur = cv2.GaussianBlur(gray_image, (5, 5), 0) # cv2.Canny already does blurring in
        canny = cv2.Canny(gray_image, 70, 150)

        if self.show_additional_windows:
            cv2.imshow("canny", canny)

        return canny

    def _display_lines(self, image, left_line, right_line, central_line):
        line_image = np.zeros_like(image)
        roi_image = np.zeros_like(image)
        # TODO: sometimes we get error for invalid type of pt1/pt2
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=10)

        x1, y1, x2, y2 = central_line
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 100), thickness=10)

        cv2.fillPoly(roi_image, pts=self.state.last_shape, color=(0, 255, 0))
        line_image = cv2.addWeighted(line_image, 1, roi_image, 0.2, 1)

        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combo_image

    def _resize(self, image):
        dim = (self.width, self.height)
        output = cv2.resize(image, dim)
        return output

    def _new_shape(self, first_line, second_line=None):
        x1, y1, x2, y2 = first_line
        offset = int(self.width * 0.1)
        ld_x = max(0, x1 - offset)
        lu_x = max(0, x2 - offset)
        rd_x = min(self.width, x1 + offset)
        ru_x = min(self.width, x2 + offset)
        if second_line is None:
            return np.array(
                [[(ld_x, y1), (lu_x, y2), (ru_x, y2), (rd_x, y1)]]
            )
        else:
            xx1, yy1, xx2, yy2 = second_line
            ld_xx = max(0, xx1 - offset)
            lu_xx = max(0, xx2 - offset)
            rd_xx = min(self.width, xx1 + offset)
            ru_xx = min(self.width, xx2 + offset)
            return np.array(
                [[(ld_x, y1), (lu_x, y2), (ru_x, y2), (rd_x, y1)],
                 [(ld_xx, yy1), (lu_xx, yy2), (ru_xx, yy2), (rd_xx, yy1)]]
            )

    def _make_coordinates(self, param):
        k, n = param
        y1 = self.height  # starts at the bottom of the image
        y2 = int(y1 / 2)  # finishes above y1

        # using formula y = kx + n
        x1 = int((y1 - n) / k)
        x2 = int((y2 - n) / k)
        return np.array([x1, y1, x2, y2])
