import rawpy
import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def convert_uint16_to_uint8(img):
    img_new = (img.astype(np.float32)*4)/2**8
    return np.uint8(img_new)

def load_data(folder,numImages=200):
    """
    this function should load your images as a numpy array

    The images are stored as "Raw" files. There are many ways how to load a raw images.
    Do a google research, there are many resources that will help you.
    
    All information that you need for the images hrer are the following:
    
    Size of images: (600 x 800)
    Dtype of Raw images: np.uint8
    
    
    args:
        folder: Only the foldername inside 'pic//HW2_data//. Should not include the complete path!
        numImages: number of images you want to load (might come in handy for debugging your code)
          
    returns:
        imgs (np.array): The 200 images as monochromatic images in uint8 type format
    """
    raise NotImplementedError

def load_dataset():
    """
    Should load all data into 2 large numpy arrays
    
    You probably need about 2-3 GB of free RAM on your computer
    
    The sensitivies in the folders are [0,1,3,9,14,18]
    
    returns:
        dark (np.array): A 4-dimensional numpy array with dim = (600, 800, 200, 6) for the dark images
        imgs (np.array): A 4-dimensional numpy array with dim = (600, 800, 200, 6) for the white images
        sensitivy (np.array): A numpy array containing [0,1,3,9,14,18]
    
    """
    raise NotImplementedError