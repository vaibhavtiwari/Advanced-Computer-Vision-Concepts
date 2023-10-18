import cv2 as cv
import numpy as np
import os
from .admin import output_folder, input_folder


class HDRImaging:
    def __init__(self):
        self.path = os.path.join(input_folder, "HDRImagesInput")

    def loadExposureSeq(self, path):
        images = []
        times = []
        with open(os.path.join(path, 'list.txt')) as f:
            content = f.readlines()
        for line in content:
            tokens = line.split()
            images.append(cv.imread(os.path.join(path, tokens[0])))
            times.append(1 / float(tokens[1]))
        return images, np.asarray(times, dtype=np.float32)

    def convert(self):
        #Firstly we load input images and exposure times from user-defined folder. The folder should contain images and list.txt -
        # file that contains file names and inverse exposure times.
        images, times = self.loadExposureSeq(self.path)
        # It is necessary to know camera response function (CRF) for a lot of HDR construction algorithms. We use one of the calibration algorithms
        # to estimate inverse CRF for all 256 pixel values.
        #We use Debevec's weighting scheme to construct HDR image using response calculated in the previous item.
        calibrate = cv.createCalibrateDebevec()
        response = calibrate.process(images, times)
        merge_debevec = cv.createMergeDebevec()
        hdr = merge_debevec.process(images, times, response)

        # Since we want to see our results on common LDR display we have to map our HDR image to 8-bit range preserving most details.
        # It is the main goal of tonemapping methods. We use tonemapper with bilateral filtering and set 2.2 as the value for gamma correction.
        tonemap = cv.createTonemap(2.2)
        ldr = tonemap.process(hdr)

        merge_mertens = cv.createMergeMertens()
        fusion = merge_mertens.process(images)

        cv.imwrite(os.path.join(output_folder, 'fusion.png'), fusion * 255)
        cv.imwrite(os.path.join(output_folder, 'ldr.png'), ldr * 255)
        cv.imwrite(os.path.join(output_folder, 'hdr.hdr'), hdr)
