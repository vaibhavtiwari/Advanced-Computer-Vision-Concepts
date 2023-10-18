from ML_Pipeline.BackGround_Substraction import BackGroundSubtraction
from ML_Pipeline.meanshift import MeanShift
from ML_Pipeline.camshift import CamShift
from ML_Pipeline.Lucal_Kanade_OpticalFlow import LucasKanadeOpticalFlow
from ML_Pipeline.DenseOpticalFlow import DenseOpticalFlow
from ML_Pipeline.HDRImages import HDRImaging
from ML_Pipeline.EpipolarGeometry import EpipolarGeometry
from ML_Pipeline.DepthMapforStereo import DepthMapStereo
from ML_Pipeline.ColorQuantization import ColorQuantization
from ML_Pipeline.ImageDenoising import ImageDenoising
from ML_Pipeline.admin import input_folder
import os


### 1. BackGround Subtraction ###
input_video = os.path.join(input_folder,'videoplayback.mp4')
#input_video = os.path.join(input_folder,'RobotinoFinal.mp4')
#background_obj = BackGroundSubtraction(input_video)
#background_obj.background_substract()


### 2. MeanShift ###
#meanshift_obj = MeanShift(input_video)
#meanshift_obj.detect(mode = "camera")
#
#
### 3. CamShift ###
#camshift_obj=CamShift(input_video)
#camshift_obj.detect(mode = "camera") #There are two modes camera and recording
#
#
#### 4. Lucas Kanade Optical Flow ###
#lucal_optical_flow_obj=LucasKanadeOpticalFlow(input_video)
#lucal_optical_flow_obj.detect()
#
#
#### 5. Farneback Dense Optical Flow ###
#dense_optical_flow_obj = DenseOpticalFlow(input_video)
#dense_optical_flow_obj.detect()
#
#
#### 6. High Dynamic Range(HDR) Imaging ###
#hdr_imaging = HDRImaging()
#hdr_imaging.convert()
#
#
#### 7. Epipolar Geometry ###
left_image = os.path.join(input_folder, "view0.png")
right_image = os.path.join(input_folder, "view2.png")
epipolar_geo_obj = EpipolarGeometry(left_image,right_image)
epipolar_geo_obj.detect()
#
#
#### 8. Depth Map on Stereo Images ###
#depth_map_obj=DepthMapStereo(left_image,right_image)
#depth_map_obj.depth()
#
#
#### 9. Color Quantization ###
#image_path = os.path.join(input_folder, "download.jpeg")
#color_quantize_obj = ColorQuantization(image_path)
#color_quantize_obj.quantize()
#
#
#### 10. Image Denoising ###
#image_denoising_object = ImageDenoising(left_image, input_video)
#image_denoising_object.denoisingcolored()
#image_denoising_object.denoising_grayscale()
#image_denoising_object.denoising_multi()