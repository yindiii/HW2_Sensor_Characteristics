import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def crop(imgs, xDimMin, xDimMax, yDimMin, yDimMax):
    """
    this crops the image defined by the following arguments
    
    Hint: This should be a one liner ! DO NOT USE for loop
    
    args:
        imgs(np.ndarray): numpy array of all images to be cropped
        xDimMin(int): this is the x Dimension minimum
        xDimMax(int): this is the x Dimension maximum
        yDimMin(int): this is the y Dimension minimum
        yDimMax(int): this is the y Dimension maximum
        
    returns:
        images (np.ndarray): cropped version of all the images
    """
    return imgs[yDimMin:yDimMax, xDimMin:xDimMax]

def channel_filter(imgs):
    """
    this filters the image to a specific channel of the bayers pattern.

    BAYER PATTERN:
    Here is a wikipedia page of the Bayer Filter:
        https://en.wikipedia.org/wiki/Bayer_filter
    
    Quick runthrough, what the Bayer pattern/filter is arranging 3 colors, green; red; blue,
    in a pattern to filter light which allows for the cameras to capture the colored images.
    The pattern contains 50% green, 25% red, and 25% blue. So what this channel filter function
    does is breaks an image into the separate filters created by the camera, and we can do
    analysis on each of these filters!
    
    Hint: You have 50% green pixels. Just take every second here to keep the same image dimension
    as for the R and B channel

    EXAMPLE:
    If we look at the bayer pattern, every second pixel for every other row is red, so we can say
    the x dimension be shifted by 1 over to be able to look at every red pixel, and look at every
    other row.
    
    args:
        imgs(np.ndarray): numpy array of all images to be filtered
    
    returns:
        filteredImages(np.ndarray): filtered version of the image, whose dimensions are 
        (X,Y, RGB, timestamp of image)
    """
    height, width, num_images = imgs.shape
    red_channel = np.zeros_like(imgs)
    green_channel = np.zeros_like(imgs)
    blue_channel = np.zeros_like(imgs)
    red_channel[1::2, 0::2] = imgs[1::2, 0::2] 
    green_channel[0::2, 0::2] = imgs[0::2, 0::2]  
    green_channel[1::2, 1::2] = imgs[1::2, 1::2] 
    blue_channel[0::2, 1::2] = imgs[0::2, 1::2]
    filteredImages = np.stack((red_channel, green_channel, blue_channel), axis=2) 
    return filteredImages
