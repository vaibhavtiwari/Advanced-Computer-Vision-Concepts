import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from .admin import input_folder


class EpipolarGeometry:
    """https://vision.middlebury.edu/stereo/data/scenes2005/"""

    def __init__(self,left_image,right_image):
        self.left_image = left_image
        self.right_image = right_image

    def drawlines(self, img1, img2, lines, pts1, pts2):
        # this function will draw lines over our images which can be used to visualize the epilines.
        """ img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines """
    
        r, c = img1.shape
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR) # convert gray image to bgr
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR) # convert gray image to bgr
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist()) # 3 random combination between 0 to 255
            x0, y0 = map(int, [0, -r[2] / r[1]])                 #starting cordinates 
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])    # end point cordinates
            img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)   # draw lines on the images
            img1 = cv.circle(img1, tuple(pt1), 5, color, -1)     # draw circle on the images
            img2 = cv.circle(img2, tuple(pt2), 5, color, -1)     # draw circle on the images
        return img1, img2

    def detect(self):

        img1 = cv.imread(self.left_image, 0)
        img2 = cv.imread(self.right_image, 0)

        sift = cv.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper - It compares the distance between two nearest neighbors for identifying distinctive correspondence
        # The idea is that for a good match the distance between two nearest matches for a certain keypoint is below a threshold, or vice versa.
        # epipolar geometry algorithm is used to find if two images are same. 

        for i, (m, n) in enumerate(matches): 
            if m.distance < 0.8 * n.distance: # checking for the distance between the two images
                pts2.append(kp2[m.trainIdx].pt) # train image - img2 / right image
                pts1.append(kp1[m.queryIdx].pt) # query image - img1 / left image
        #This loop defines which matches will be discarded and which matches will be preserved.
        #So, if m distance is less than 80% of n distance, then the descriptor for that particular row will be preserved. 
        #On the other hand, if m distance is greater than 80% of n distance, then it’s probably not a good match, and the descriptor for
        #that particular row will be discarded. Therefore less distance means a better match.

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = self.drawlines(img1, img2, lines1, pts1, pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.drawlines(img2, img1, lines2, pts2, pts1)
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.show()
