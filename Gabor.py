
# Gabor filter - 
# run Gabor.py --images Gab_Images


from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
for imagePath in paths.list_images(args["images"]):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	img = cv2.imread(imagePath)
#img = cv2.imread(img)
    #img=cv2.imread('C:/Users/lavan/Gab_Images/Lena.jpg')
	img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	df=pd.DataFrame()
	img2=img.reshape(-1)
	df['original pixels']=img2
    #ksize=5
	num=1
	for theta in range(2):
		theta=theta/4*np.pi
		for sigma in (3,5):
			for lamda in np.arange (0,np.pi,np.pi/8.):
				for gamma in(0.05, 0.5):
                #print(theta,sigma,lamda,gamma)
					gabor_label='Gabor'+ str(num)
					kernal=cv2.getGaborKernel((5,5),sigma, theta,lamda, gamma,0,ktype= cv2.CV_32F)
					fimg=cv2.filter2D(img, cv2.CV_8UC3, kernal)
					filtered_img=fimg.reshape(-1)
					df[ gabor_label]=filtered_img
					num+=1
					df.to_csv('Gabor3.csv')
#print(gabor_label)
#print(df.head())
#df.to_csv('Gabor3.csv')
#					kernal_resized=cv2.resize(kernal,(400,400))
#					cv2.imshow('Kernal',kernal_resized)
#					cv2.imshow('Originalimage',img)
#					cv2.imshow('Filtered',fimg)
#					cv2.waitKey(0)
#					cv2.destroyAllWindows()
