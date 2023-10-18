from __future__ import print_function
import cv2 as cv



class BackGroundSubtraction:
    '''https://stackoverflow.com/questions/33266239/differences-between-mog-mog2-and-gmg'''

    def __init__(self,input_video_path):

        self.input_video = input_video_path

    def background_substract(self, algo=""):

        if algo == 'MOG2':
            """Gaussian Mixture-based Background/Foreground Segmentation Algorithm
            It uses a method to model each background pixel by a mixture of K Gaussian distributions
            """
            backSub = cv.createBackgroundSubtractorMOG2()
        else:
            """history	Length of the history.
                dist2Threshold	Threshold on the squared distance between the pixel and the sample to decide whether a pixel is close to that sample. This parameter does not affect the background update.
                detectShadows	If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false. """
            backSub = cv.createBackgroundSubtractorKNN()
            
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(self.input_video))
        if not capture.isOpened:
            print('Unable to open: ' + self.input_video)
            exit(0)
        while True:
            ret, frame = capture.read()
            if frame is None:
                break

            fgMask = backSub.apply(frame)

            cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)

            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
