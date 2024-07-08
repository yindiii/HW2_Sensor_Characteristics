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

def load_data(folder,numImages=200,height=600,width=800):
    """
    this function should load your images as a numpy array

    The images are stored as "Raw" files. There are many ways how to load a raw images.
    Do a google research, there are many resources that will help you.
    
    You can e.g. use np.fromfile. However, this gives only a 1D-array back, hence you have to
    resize it after
    
    All information that you need for the images hrer are the following:
    
    Size of images that are used in this assignemnt are: 600 x 800
    Dtype of Raw images: np.uint8
    
    You can use e.g. the glob function with the wildcard (*) parameter to get a list of
    only the raw files. Then you only have to iterate through this list and load each image and
    store it.
    
    
    args:
        folder: Only the foldername inside 'pic//HW2_data//. Should not include the complete path!
        numImages: number of images you want to load (might come in handy for debugging your code)
        height: height of the image in pixels
        width: width of the images in pixels
          
    returns:
        imgs (np.array): The 200 images as monochromatic images in uint8 type format
    """

    path = os.path.join('pic', 'HW2_data', folder, '*.raw')
    raw_files = glob.glob(path)
    #print(f"Found {len(raw_files)} raw files.")
    
    # Initialize an empty array to hold images
    imgs = np.empty((numImages, height, width), dtype=np.uint8)
    
    # Load each raw image file
    for i, file in enumerate(raw_files[:numImages]):
        img = np.fromfile(file, dtype=np.uint8)
        img = img.reshape((height, width))
        imgs[i] = img
    
    imgs = np.transpose(imgs, (1, 2, 0))
    
    return imgs

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
    sensitivities = [0, 1, 3, 9, 14, 18]
    dark_images = []
    white_images = []
    
    for sensitivity in sensitivities:
        dark_folder = f'dark{sensitivity}'
        white_folder = f'gain{sensitivity}'
        dark_imgs = load_data(dark_folder)
        dark_images.append(dark_imgs)
        white_imgs = load_data(white_folder)
        white_images.append(white_imgs)
    dark_images = np.array(dark_images)
    white_images = np.array(white_images)
    dark_images = np.stack(dark_images, axis=-1).transpose((0, 1, 2, 3))
    white_images = np.stack(white_images, axis=-1).transpose((0, 1, 2, 3))
    
    return dark_images, white_images, np.array(sensitivities)

def display_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()