import os
import numpy as np
import cv2
import math
import SimpleITK as sitk
import skimage.io as io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_img(path):
	img = sitk.ReadImage(path)
	data = sitk.GetArrayFromImage(img)
	return data

#show_img(img[i,:,:])
def show_img(img):
	io.imshow(img,cmap = 'gray')
	io.show()

def plt_show(img,i):
	plt.subplot(1,2,i)
	plt.imshow(img,cmap='gray')
		
def imgaussian(I,sigma):
	siz = sigma*6
	if sigma>0:
		x = range(-int(math.ceil(siz/2)),int(math.ceil(siz/2)+1))
		print type(np.array(x)),type(sigma)
		H = np.exp(-(x**2/(2*sigma**2)))
		H = H/np.sum(H)
		print H

def Hessian2D(img,sigma):
	#don't need to make sigma a non-integer coz they have no difference
	x = range(int(-round(3*sigma)),int(round(3*sigma)+1))
	[X,Y] = np.meshgrid(x,x)

	expX = -(X**2+Y**2)/(2*(sigma**2))
	s = np.shape(expX)
	for i in range(0,s[0]):
		for j in range(0,s[1]):
			expX[i][j] = np.exp(expX[i][j])

	DGaussxx = np.dot(1/(2*math.pi*(sigma**4))*(X**2/(sigma**2)-1),expX)
	DGaussxy = np.dot(1/(2*math.pi*sigma**6)*(np.dot(X,Y)),expX)
	DGaussyy = np.transpose(DGaussxx)

	Dxx = cv2.filter2D(img,-1,DGaussxx)
	Dxy = cv2.filter2D(img,-1,DGaussxy)
	Dyy = cv2.filter2D(img,-1,DGaussyy)

	return Dxx,Dxy,Dyy

def eig2image(Dxx,Dxy,Dyy):
	tmp = np.sqrt((Dxx-Dyy)**2+4*Dxy**2)
	v2x = 2*Dxy
	v2y = Dyy-Dxx+tmp
	
	#Normalize
	mag = np.sqrt(v2x**2+v2y**2)
	i = np.array(mag!=0)
	v2x[i==True] = v2x[i==True]/mag[i==True]
    	v2y[i==True] = v2y[i==True]/mag[i==True]

	v1x = -v2y
	v1y = v2x

	mu1 = 0.5*(Dxx+Dyy+tmp)
	mu2 = 0.5*(Dxx+Dyy-tmp)

    	check=abs(mu1)>abs(mu2)
            
    	Lambda1=mu1.copy()
    	Lambda1[check==True] = mu2[check==True]
    	Lambda2=mu2
    	Lambda2[check==True] = mu1[check==True]
    
    	Ix=v1x
    	Ix[check==True] = v2x[check==True]
    	Iy=v1y
    	Iy[check==True] = v2y[check==True]
	
	return Lambda1,Lambda2,Ix,Iy

	

def FrangiFilter2D(img,beta,c):
	FrangiScaleRatio = 2
	FrangiScaleRange = range(1,10,FrangiScaleRatio)
	FrangiBetaOne = beta
	FrangiBetaTwo = c
	verbose = 0
	BlackWhite = 0
	sigmas = np.array(range(FrangiScaleRange[0],FrangiScaleRange[1]+1))
	beta = 2*FrangiBetaOne**2
	c = 2*FrangiBetaTwo**2

	ALLfiltered = np.zeros(np.shape(img)+ (len(sigmas),))
	ALLangles = np.zeros(np.shape(img)+ (len(sigmas),))
	
	for i in range(0,len(sigmas)):
		if verbose == 1:
			print('Current Frangi Filter Sigma:',sigmas[i])
		
		#make 2D hessian
		[Dxx,Dxy,Dyy] = Hessian2D(img,sigmas[i])
		Dxx = (sigmas[i]**2)*Dxx;
		Dxy = (sigmas[i]**2)*Dxy;
		Dyy = (sigmas[i]**2)*Dyy;

		#Calculate
		[Lambda2,Lambda1,Ix,Iy] = eig2image(Dxx,Dxy,Dyy)
		
		si = np.shape(Ix)
		angles = np.zeros(si)
		for j in range(0,si[0]):
			for k in range(0,si[1]):
				angles[j][k] = math.atan2(Ix[j][k],Iy[j][k])
		
		sl = np.shape(Lambda1)
		Rb = np.zeros(sl)
		S2 = np.zeros(sl)
		Ifiltered = np.zeros(sl)
		for j in range(0,sl[0]):
			for k in range(0,sl[1]):
				if Lambda1[j][k] == 0:
					Lambda1[j][k] = np.spacing(1)
				Rb[j][k] = (Lambda2[j][k]/Lambda1[j][k])**2
				S2[j][k] = Lambda1[j][k]**2+Lambda2[j][k]**2
				A = np.exp(-Rb[j][k]/beta)
				B = 1-np.exp(-S2[j][k]/c)
				#print ('Lambda1:'+str(Lambda1[j][k])+'   Lambda2:'+str(Lambda2[j][k]))
				#print ('A:'+str(A)+'   B:'+str(B))
				Ifiltered[j][k] = np.exp(-Rb[j][k]/beta)*(1-np.exp(-S2[j][k]/c))
				maxi = np.max(Ifiltered)
				mini = np.min(Ifiltered)
				if BlackWhite == 1:
					if Lambda1[j][k] < 0:
						Ifiltered[j][k] = 0
				else:
					if Lambda1[j][k] > 0:
						Ifiltered[j][k] = 0


		Ifiltered = (Ifiltered-np.min(Ifiltered))*255/(np.max(Ifiltered)-np.min(Ifiltered))
		ALLfiltered[:,:,i] = Ifiltered
		ALLangles[:,:,i] = angles

	if len(sigmas)>1:
		outIm=np.max(ALLfiltered,2)
		
	else:
		outIm = (outIm.transpose()).reshape(np.shape(img))
						
	return outIm
		



#test image
#filename = "vessel.png"
#img = mpimg.imread(filename)
#shape = np.shape(img)
#img_blur = np.zeros(shape)
#beta = 0.5#range=[0.3,2]
#c = 15 #range=[10^-5,10^-6]
#show_img(img)
#filteredimg = FrangiFilter2D(img,beta,c)
#show_img(filteredimg)

#img = cv2.GaussianBlur(img[],)


filename = ""#input your file
img = read_img(filename)
beta = 1.5#range=[0.3,2]
c = 0.000001 #range=[10^-5,10^-6]
shape = np.shape(img)
simg = np.shape(img)
smax = np.max(img)
smin = np.min(img)
for j in range(0,simg[0]):
	for k in range(0,simg[1]):
		img[j][k] = round(img[j][k]*255/smax)
img_blur = np.zeros(shape)
img_blur = cv2.GaussianBlur(img,(3,3),2)
plt.figure()
plt_show(img_blur,1)

#Normalization
#smax = np.max(img_blur)
#smin = np.min(img_blur)
#for j in range(0,simg[0]):
#	for k in range(0,simg[1]):
#		img_blur[j][k] =  (img_blur[j][k]-smin)/(smax-smin)

filteredimg = FrangiFilter2D(img_blur,beta,c)
plt_show(filteredimg,2)
plt.show()

#writer = sitk.ImageFileWriter()
#writer.SetFileName(outputImageFileName)
#writer.Execute(image)
