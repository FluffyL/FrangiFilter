import os
import numpy as np
import scipy
import cv2
import math
import pandas as pd
import SimpleITK as sitk
import skimage.io as io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_img(path):
	img = sitk.ReadImage(path)
	data = sitk.GetArrayFromImage(img)
	return data

def img_show(img):
	for i in range(img.shape[0]):
		io.imshow(img[i,:,:],cmap='gray')
		print(i)
		io.show()

def img_store(img,outpath):
	out = sitk.GetImageFromArray(img)
	sitk.WriteImage(out,outpath)
'''
def imgaussian(I,sigma):
	siz = sigma*6
	if sigma>0:
		x = np.array(range(-int(math.ceil(siz/2)),int(math.ceil(siz/2)+1)))
		H = np.exp(-(x**2/(2*sigma**2)))
		H = H/np.sum(H)
		print x
'''

def gradient3(F,option):

	k,l,m = np.shape(F)
	D = np.zeros(np.shape(F))
	if option == 'x':
		D[0,:,:] = (F[1,:,:] - F[0,:,:])
    		D[k-1,:,:] = (F[k-1,:,:] - F[k-2,:,:])
		D[1:k-2,:,:] = (F[2:k-1,:,:]-F[0:k-3,:,:])/2
	if option == 'y':
		D[:,0,:] = (F[:,1,:] - F[:,0,:])
    		D[:,l-1,:] = (F[:,l-1,:] - F[:,l-2,:])
    		D[:,1:l-2,:] = (F[:,2:l-1,:]-F[:,0:l-3,:])/2
	if option == 'z':
		D[:,:,0] = (F[:,:,0] - F[:,:,0])
    		D[:,:,m-1] = (F[:,:,m-1] - F[:,:,m-2])
    		D[:,:,1:m-2] = (F[:,:,2:m-1]-F[:,:,0:m-3])/2
	return D

def Hessian3D(Volume,Sigma=1):
	if Sigma > 0:
		F = cv2.GaussianBlur(Volume,(0,0),sigmaX=Sigma)
	else:
		F = Volume
	Dz = gradient3(F,'z')
	Dzz = gradient3(Dz,'z')
	Dy = gradient3(F,'y')
	Dyy = gradient3(Dy,'y')
	Dyz = gradient3(Dy,'z')
	Dx = gradient3(F,'z')
	Dxx = gradient3(Dx,'x')
	Dxy = gradient3(Dx,'y')
	Dxz = gradient3(Dx,'z')	
	return Dxx,Dyy,Dzz,Dxy,Dxz,Dyz

def eig3volume(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz):
	temp1 = np.array([[Dxx,Dxy,Dxz],[Dxy,Dyy,Dyz],[Dxz,Dyz,Dzz]])
	E1,E2 = np.linalg.eig(temp1)
        E = np.argsort(
        
	Lambda1 = E[0][0]
	Lambda2 = E[1][0]
	Lambda3 = E[2][0]

	if np.linalg.norm(Lambda1)<np.linalg.norm(Lambda2):
		temp2 = Lambda2
		Lambda2 = Lambda1
		Lambda1 = temp2
	if np.linalg.norm(Lambda1)<np.linalg.norm(Lambda3):
		temp2 = Lambda3
		Lambda3 = Lambda1
		Lambda1 = temp2
	if np.linalg.norm(Lambda2)<np.linalg.norm(Lambda3):
		temp2 = Lambda3
		Lambda3 = Lambda2
		Lambda2 = temp2
	
	return Lambda1,Lambda2,Lambda3

def FrangiFilter3D(I, FrangiScaleRange=[1,8], FrangiScaleRatio=2, FrangiAlpha=0.5, FrangiBeta=1.5, FrangiC=0.0000001, BlackWhite=1, verbose=1):
	sigma = range(FrangiScaleRange[0],FrangiScaleRange[1])
	sigma = np.array(sorted(sigma))

	for i in range(len(sigma)):
		if verbose == 1:
			print "Current Frangi Filter Sigma:  ",sigma[i]

		Dxx,Dyy,Dzz,Dxy,Dxz,Dyz = Hessian3D(I,sigma[i])
		if sigma[i]>0:
			c = sigma[i]^2
			Dxx = c*Dxx
			Dxy = c*Dxy
			Dxz = c*Dxz
			Dyy = c*Dyy
			Dyz = c*Dyz
			Dzz = c*Dzz
		Lambda1,Lambda2,Lambda3 = eig3volume(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz)

		LambdaAbs1=abs(Lambda1)
    		LambdaAbs2=abs(Lambda2)
    		LambdaAbs3=abs(Lambda3)
		Ra = LambdaAbs2/LambdaAbs3
		Rb = LambdaAbs1/np.sqrt(LambdaAbs2*LambdaAbs3)
	
		S = np.sqrt(LambdaAbs1**2 + LambdaAbs2**2 + LambdaAbs3**2)
		A = 2 * FrangiAlpha**2
		B = 2 * FrangiBeta**2
		C = 2 * FrangiC**2

		expRa = (1-np.exp(-(Ra**2/A)))
		expRb = np.exp(-(Rb**2/B))
		expS = (1-np.exp(-S**2/(2*FrangiC**2)))

		Voxel_data = expRa * expRb * expS
		#img_show(Voxel_data)
		if BlackWhite == 1:
			Voxel_data[Lambda2<0] = 0
			Voxel_data[Lambda3<0] = 0
		else:
			Voxel_data[Lambda2>0] = 0
			Voxel_data[Lambda3>0] = 0

		#using 0 instead of NaN values
		#print Voxel_data
		Voxel_data[pd.isnull(Voxel_data)]=0
		
		if i == 0:
			Iout = Voxel_data
			'''
			whatScale = np.ones(np.shape(I))
			Voutx = Vx
			Vouty = Vy
			Voutz = Vz
			'''
		else:
			'''
			whatScale[Voxel_data>Iout] = i;
			Voutx[Voxel_data>Iout]=Vx[Voxel_data>Iout]
			Vouty[Voxel_data>Iout]=Vy[Voxel_data>Iout]
			Voutz[Voxel_data>Iout]=Vz[Voxel_data>Iout]
			'''
			Iout[Iout<Voxel_data]=Voxel_data[Iout<Voxel_data]
	return Iout

filename = "/home/lly/Desktop/intern_Xidian/MingXi/Hessian/hessian_brain/MR00008133_skullstriped_corrected.nii.gz"
img = read_img(filename)
		
simg = np.shape(img)
smax = np.max(img)
smin = np.min(img)
for j in range(0,simg[0]):
	for k in range(0,simg[1]):
		for l in range(0,simg[2]):
			img[j][k][l] = round(img[j][k][l]*255/smax)

img = FrangiFilter3D(img)
smax = np.max(img)
smin = np.min(img)
for j in range(0,simg[0]):
	for k in range(0,simg[1]):
		for l in range(0,simg[2]):
			img[j][k][l] = round(img[j][k][l]*255/smax)
print img
print np.shape(img)
img_show(img)

	
