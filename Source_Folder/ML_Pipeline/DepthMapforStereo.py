import cv2 as cv
from matplotlib import pyplot as plt


class DepthMapStereo:
    def __init__(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image

    def depth(self):
        # reading both of the images in grayscale format
        imgL = cv.imread(self.left_image, 0)
        imgR = cv.imread(self.right_image, 0)
        """
        numDisparities the disparity search range. For each pixel algorithm will find the best
         blockSize the linear size of the blocks compared by the algorithm. The size should be odd
        """
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        plt.imshow(disparity, 'gray')
        plt.show()
