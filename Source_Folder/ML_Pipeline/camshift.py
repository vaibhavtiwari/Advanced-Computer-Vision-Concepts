import numpy as np
import cv2 as cv


class CamShift:

    def __init__(self, input_video_path):
        self.input_video = input_video_path

    def detect(self, mode="camera"):

        if mode == "camera":
            cap = cv.VideoCapture(0)
        elif mode == "recording":
            cap = cv.VideoCapture(self.input_video)

        # take first frame of the video
        ret, frame = cap.read()
        # select region of interest
        bbox = cv.selectROI(frame)
        x, y, w, h = bbox
        # x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
        track_window = (x, y, w, h)
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
        roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        while (1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply camshift to get the new location
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                # Draw it on image
                pts = cv.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv.polylines(frame, [pts], True, 255, 2)
                cv.imshow('img2', img2)

                # now update the roi coordinates to new values
                # x, y, w, h = track_window
                # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                ## display track
                # cv.imshow("CAMshift Track", frame)

                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break
