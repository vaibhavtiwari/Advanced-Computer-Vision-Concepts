# Advanced-Computer-Vision-Concepts

  1) **BackGround Subtraction**
     In OpenCV we have three algorithms to do this operation:
       1) BackgroundSubtractorMOG: It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
       2) BackgroundSubtractorMOG2: It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It provides better adaptability to varying scenes due illumination changes, etc.
       3) BackgroundSubtractorGMG: This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation.

  2) **MeanShift**
     1) Every instance of video is checked in the form of pixel distribution in that frame.
     2) The initial window can be hardcoded or selected by the user, generally a square or rectangle for which the positions are specified, and the algorithm tracks the area of maximum pixel distribution.
     3) The window keeps shifting towards the area of maximum pixel distribution.
     4) The direction of movement depends on the difference between the center of our tracking window and the centroid of all the k-pixels inside that window.
     5) CONS: Size of the tracking window remains same irrespective of distance of object from the camera.
    
  3) **CamShift (Continuously Adaptive Mean Shift)**
     1) Enhanced version of mean Shift which provides more accuracy and robustness of the model.
     2) Size of the window keeps updating when tracking windows tries to converge.
     3) Tracking is done by using the color information of the object.
     4) It actually first applies mean Shift and then updates the size of window.

  4) **Optical Flow**
     It is a task of per-pixel motion estimation between two consecutive frames in one video. It implies the calculation of the shift vector for a pixel as an object displacement difference between two neighbors. The idea is to estimate the object’s displacement vector caused by it’s motion or camera movements.

    Below are the two algorithms for optical flow:

    1) **Lucas-Kanade Algorithm**
       1) Used with sparse feature sets, computes the motion vector for the specific set of objects (detected corners on the image).
       2) Using only a sparse feature set means that we will not have motion information about pixels that are not contained in it.

    2) **Dense Optical Flow**
       1) The motion vector for every pixel in the image is calculated.
       2) The farneback algorithm requires a 1-dimensional input image,so the BRG image is converted into grayscale.
       3) In dense optical flow, the algorithm look at all of the points (unlike Lucas Kanade, which works only on corner points detected) and detects the pixel intensity changes between the two frames, resulting in an image with highlighted pixels after converting to hsv format for clear visibility.
       4) It computes the magnitude and direction of optical flow from an array of the flow vectors, i.e., (dx/dt, dy/dt). Later, it visualizes the angle (direction) of flow by hue and the distance (magnitude) of flow by value of the HSV color representation.For visibility to be optimal, strength of HSV is set to 255.


  5) **HDR Imaging (High Dynamic Range)**
     1) Most digital images and devices use 8 bits per channel,thus limiting the dynamic range of the device to two orders of magnitude.
     2) When we take photographs of a real world scene,bright regions may  be overexposed and the dark ones may be underexposed ,so we can’t capture all details using a single exposure.
     3) HDR imaging works with images that use more than 8 bits per channel.
     4) The way to take HDR images ,is to be use photographs of the scene taken with different exposure values.
     5) **Tonemapping** : After the conversion of HDR image, we need to convert that to 8-bit view to view on usual displays.That is called as Tonemapping.
