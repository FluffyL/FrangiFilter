from __future__ import print_function
from functools import reduce

import sys
import os
import numpy as np
import cv2
import math
import SimpleITK as sitk
import skimage.io as io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_img(path):
	data = sitk.ReadImage(path,sitk.sitkFloat32)
	data = sitk.GetArrayFromImage(data)
	return data


def img_show(img):
	for i in range(img.shape[0]):
		io.imshow(img[i,:,:],cmap='gray')
		print(i)
		io.show()


def img_store(img,outpath):
	out = sitk.GetImageFromArray(img)
	sitk.WriteImage(out,outpath)


def binary_mask(img):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			for k in range(img.shape[2]):
				if img[i,j,k] > 0:
					img[i,j,k] = 255
	return img


def biasFieldCorrection(img,shrink):
	numberFittingLevels = 4
	inputImage = sitk.ReadImage(img,sitk.sitkFloat32)
	maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
	inputImage = sitk.Shrink(inputImage,[int(shrink)]* inputImage.GetDimension())
	maskImage = sitk.Shrink(maskImage, [int(shrink)] * inputImage.GetDimension() )
	inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
	corrector = sitk.N4BiasFieldCorrectionImageFilter();
	output = corrector.Execute(inputImage, maskImage)

	return output

#delete labeled voxel(skull strip)
'''
path1 = '.nii.gz'#your file
img1 = read_img(path1)
path2 = '.nii.gz'#your label
img2 = read_img(path2)
outpath = 'MR00008133_skullstriped.nii.gz'
for i in range(img1.shape[0]):
	for j in range(img1.shape[1]):
		for k in range(img1.shape[2]):
			if img2[i,j,k] > 0:
				img1[i,j,k]=0
img_show(img1)
img_store(img1,outpath)
'''


#biasFieldCorrection

outpath = '.nii.gz'#corrected file
path = '.nii.gz'#skull striped file
#img = read_img(path)
#print(np.shape(img))
shrink = 1
output = biasFieldCorrection(path,shrink)
output = sitk.GetArrayFromImage(output)
img_show(output)
img_store(output,outpath)

