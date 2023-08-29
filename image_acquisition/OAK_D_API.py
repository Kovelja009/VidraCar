import cv2
import depthai as dai
import numpy as np
import time


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

        self._coordinates = (30, 50)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._color = (0, 0, 255)
        self._thickness = 2

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

    def show_fps(self, frame, fps):
        return cv2.putText(frame, fps.__str__(), self._coordinates, self._font, self._font_scale, self._color,
                           self._thickness, cv2.LINE_AA)


class OAK_D:
    def __init__(self, fps=24, width=720, height=480, extended_disparity=False, subpixel=False, lr_check=True):
        # Create pipeline
        self._pipeline = dai.Pipeline()

        # Define source and output
        self._camRgb = self._pipeline.create(dai.node.ColorCamera)
        self._xoutVideo = self._pipeline.create(dai.node.XLinkOut)
        # Define sources and outputs
        self._monoLeft = self._pipeline.create(dai.node.MonoCamera)
        self._monoRight = self._pipeline.create(dai.node.MonoCamera)
        self._depth = self._pipeline.create(dai.node.StereoDepth)
        self._xoutDisparity = self._pipeline.create(dai.node.XLinkOut)

        self._xoutVideo.setStreamName("video")
        self._xoutDisparity.setStreamName("disparity")

        # Properties
        self._camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self._camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self._camRgb.setInterleaved(False)
        self._camRgb.setVideoSize(width, height)
        self._camRgb.setFps(fps)

        self._monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self._monoLeft.setCamera("left")
        self._monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self._monoRight.setCamera("right")

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        self._extended_disparity = extended_disparity
        # Better accuracy for longer distance, fractional disparity 32-levels:
        self._subpixel = subpixel
        # Better handling for occlusions:
        self._lr_check = lr_check

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        self._depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        self._depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self._depth.setLeftRightCheck(lr_check)
        self._depth.setExtendedDisparity(extended_disparity)
        self._depth.setSubpixel(subpixel)

        self.blocking = self._xoutVideo.input.setBlocking(False)
        self._xoutVideo.input.setQueueSize(1)

        # Linking
        self._camRgb.video.link(self._xoutVideo.input)

        self._monoLeft.out.link(self._depth.left)
        self._monoRight.out.link(self._depth.right)
        self._depth.disparity.link(self._xoutDisparity.input)

        # Connect to device and start pipeline
        self._device = dai.Device(self._pipeline)
        self._video = self._device.getOutputQueue(name="video", maxSize=1, blocking=False)
        # Output queue will be used to get the disparity frames from the outputs defined above
        self._disparity_q = self._device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

        self.fps_handler = FPSHandler()
        self.height = self._camRgb.getVideoHeight()
        self.width = self._camRgb.getVideoWidth()

    def get_color_frame(self, show_fps=False):
        video_in = self._video.get()
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        if show_fps:
            self.fps_handler.next_iter()
            # return video_in.getCvFrame()
            return self.fps_handler.show_fps(video_in.getCvFrame(), round(self.fps_handler.fps(), 2))
        else:
            return video_in.getCvFrame()

    def get_disparity_frame(self, is_color=False):
        inDisparity = self._disparity_q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / self._depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        if is_color:
            # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        return frame


if __name__ == '__main__':
    oak_d = OAK_D()
    while True:
        color_frame = oak_d.get_color_frame(show_fps=True)
        disparity_frame = oak_d.get_disparity_frame(is_color=False)
        cv2.imshow("color", color_frame)
        cv2.imshow("disparity", disparity_frame)
        if cv2.waitKey(1) == ord('q'):
            break
