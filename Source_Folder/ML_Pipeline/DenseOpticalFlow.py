import numpy as np
import cv2 as cv


class DenseOpticalFlow:
    """http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf"""

    def __init__(self, input_video_path):
        self.input_video = input_video_path

    def detect(self):
        cap = cv.VideoCapture(self.input_video)
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        while True:
            ret, frame2 = cap.read()
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            """
            prev	first 8-bit single-channel input image.
            next	second input image of the same size and the same type as prev.
            flow	computed flow image that has the same size as prev and type CV_32FC2.
            pyr_scale	parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
            levels	number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
            winsize	averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
            iterations	number of iterations the algorithm does at each pyramid level.
            poly_n	size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
            poly_sigma	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
            flags	operation flags that can be a combination of the following:
            
                OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
                OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
            """
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame2)
                cv.imwrite('opticalhsv.png', bgr)
            prvs = next
