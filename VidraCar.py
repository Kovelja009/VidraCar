from image_acquisition.OAK_D_API import OAK_D
from steering_angle.angle_prediction import AngleCalculator
from obstacle_detection.distance_awareness import DistanceAlerter
import cv2

if __name__ == '__main__':
    # pipeline for interacting with OAK-D camera
    oak_d = OAK_D()
    angle_calc = AngleCalculator(height=480, width=720, resize=0.4, decay=0.7, draw_lines=True)
    distance_alerter = DistanceAlerter(width=640, height=400, roi_width=320, roi_height=300, distance_threshold=190,
                                       size_treshold=0.2)

    while True:
        color_frame = oak_d.get_color_frame(show_fps=True)
        computed_frame, angle = angle_calc.get_angle(color_frame)
        disparity_frame = oak_d.get_disparity_frame(is_color=False)
        distance_frame, should_stop = distance_alerter.check_distance(disparity_frame)

        cv2.imshow("angle", computed_frame)
        cv2.imshow("disparity", distance_frame)
        if cv2.waitKey(1) == ord('q'):
            break
