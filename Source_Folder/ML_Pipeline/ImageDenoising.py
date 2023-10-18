
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class ImageDenoising:
    """
    h : parameter deciding filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
    hForColorComponents : same as h, but for color images only. (normally same as h)
    templateWindowSize : should be odd. (recommended 7)
    searchWindowSize : should be odd. (recommended 21)
    """

    def __init__(self, image_path, video_path):
        self.image = image_path
        self.video_path = video_path

    def denoisingcolored(self):
        img = cv.imread(self.image)
        # common parameters are discussed above
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(dst)
        plt.show()

    def denoising_grayscale(self):
        img = cv.imread(self.image)
        # common parameters are discussed above
        dst = cv.fastNlMeansDenoising(img, None, 10, 10, 7)
        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(dst)
        plt.show()

    def denoising_multi(self):
        cap = cv.VideoCapture(self.video_path)
        # create a list of first 5 frames
        img = [cap.read()[1] for i in range(5)]
        # convert all to grayscale
        gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
        # convert all to float64
        gray = [np.float64(i) for i in gray]
        # create a noise of variance 25
        noise = np.random.randn(*gray[1].shape) * 10
        # Add this noise to images
        noisy = [i + noise for i in gray]
        # Convert back to uint8
        noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]
        # Denoise 3rd frame considering all the 5 frames
        dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

        plt.subplot(131), plt.imshow(gray[2], 'gray')
        plt.subplot(132), plt.imshow(noisy[2], 'gray')
        plt.subplot(133), plt.imshow(dst, 'gray')
        plt.show()
