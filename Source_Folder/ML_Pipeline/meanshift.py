import numpy as np
import cv2 as cv


class MeanShift:
    def __init__(self, input_video_path):
        self.input_video = input_video_path

    def detect(self, mode="camera"):

        if mode == "camera":
            cap = cv.VideoCapture(0)
        elif mode == "recording":
            cap = cv.VideoCapture(self.input_video)

        # take first frame of the video
        ret, frame = cap.read()
        # setup initial location of window
        # select region of interest
        bbox = cv.selectROI(frame)
        x, y, w, h = bbox
        track_window = (x, y, w, h)
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        while (1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply meanshift to get the new location
                ret, track_window = cv.meanShift(dst, track_window, term_crit)
                # Draw it on image
                x, y, w, h = track_window
                img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                cv.imshow('img2', img2)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break
