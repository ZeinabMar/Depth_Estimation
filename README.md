# Depth_Estimation
Estimating the depth of a scene from single
monocular color images is a fundamental problem in image
understanding. Depth estimation is a critical function for
robotic tasks such as localization, mapping, and obstacle
detection. Recent works based on the development of deep
convolutional neural networks provide reliable results. Due to
the low cost and relatively small size of monocular cameras, the
performance of neural networks for depth estimation from
single RGB image has increased significantly. Inspector robots
move inside the sewer pipe in an ambiguous environment that
has various contaminations and obstacles. As a result of
understanding the environment inside the pipe and analyzing
the images from the monocular camera, the robot can move
more safely and perform the mission more effectively. This
paper presents a new deep neural network, called SepiDepthASPP. Our approach uses the integration of ASPP and adaptive
bins to extract strong global and local contextual features at
multiple scales, and then translate them to higher resolutions for
clearer depth maps. This network is specially designed for
images inside the sewer pipe to more accurately estimate the
details of the images in the depth map.This network runs on the
dataset inside the sewer pipe and helps the robot comprehend
the inside of the pipe environment. 

# Implementation of This Paper 
AdaBins-ASPP is trained on Sewer Pipes Imagines and can been applied by below:
