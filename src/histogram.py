import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def plot_overlayed_hist(data,loc,sensitivity,size):
    """
    plot the histogram (PDF) of pixel intensities for each sensitivity setting on the camera.
    
    Hint: How to make nice histograms ?!
    1. Use the plt.his function
    2. Use alpha around 0.8
    3. you can use the option ec="k"
    4. You can use density=True
    
    args:
        data(np.ndarray): (H, W, #colors, #images, #sensitivity) array of captured pixel intensities
        loc(np.ndarray): (y,x) 2D location of pixel to plot histogram
        sensitivity(np.ndarray): (#sensitivity) array of camera sensitivity settings
        size(np.ndarray): (h,w) of pixels to include in histogram 
        
    output:
        void, but you should plot the graphs! hint: try looking at plt.hist
    """
    y, x = loc
    h, w = size

    # Ensure the region to be extracted is within bounds
    y_end = min(y + h, data.shape[0])
    x_end = min(x + w, data.shape[1])
    
    region = data[y:y_end, x:x_end, :, :, :]

    plt.figure(figsize=(10, 6))
    for i, sens in enumerate(sensitivity):
        pixel_values = region[:, :, :, :, i].flatten()
        plt.hist(pixel_values, bins=256, alpha=0.8, ec="k", density=True, label=f'Sensitivity {sens}')
    plt.title('Overlayed histogram of pixel intensities')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def get_pixel_location(img_shape,N_x,N_y):
    """
    
    Takes the shape of an image and number of to be gridded points in X and Y direction 
    to sample equally spaced points on the 2D-grid
    
    We want to exclude points at the boundaries.
    
    E.g., if our image has 100 x 100 and we want to sample 3 x 4 points we would do the following 
    
    25 50 75 for the x-coordinate
    and
    20 40 60 80 for the y-coordinate
    
    Those 2 vectors then need to converted into 2 matrices for X and Y positions (use meshgrid)
    
    the following numpy functions can come in handy to develop this function:
    
    1. np.arange
    2. np.meshgrid
    3. np.round to cast to integer values 
    4. np.astype(np.uint16) as we want integer values for the coordinates
    
    Input:
    
    Output:
    
    """
    height, width = img_shape[:2]
    x_coords = np.linspace(start=width // (N_x + 1), stop=width - width // (N_x + 1), num=N_x)
    y_coords = np.linspace(start=height // (N_y + 1), stop=height - height // (N_y + 1), num=N_y)
    x_coords = np.round(x_coords).astype(np.uint16)
    y_coords = np.round(y_coords).astype(np.uint16)
    X, Y = np.meshgrid(x_coords, y_coords)
    return X, Y