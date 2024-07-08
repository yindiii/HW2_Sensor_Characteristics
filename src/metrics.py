import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def calc_mean(imgs):
    """
    calculates the mean across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos means you are trying to 
        calculate.
    output:
        mean_imgs(np.ndarray): the mean value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """
    mean_imgs = np.mean(imgs, axis=3)
    return mean_imgs
    

def calc_var(imgs):
    """
    calculates the variance across all time stamps of the images with a specific filter
    args:
        imgs(np.ndarray): the images separated into rgb vals, whos variance you are trying to 
        calculate.
    output:
        var_imgs(np.ndarray): the variance value of images in relation to their bayer pattern
        filters. size should be (x dimension * y dimension * r g b)
    """

    var_imgs = np.var(imgs, axis=3)
    return var_imgs

def fit_linear_polynom_to_variance_mean(mean, var,th=200):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_channel, Num_gain)
    
    output:
          gain(nd.array): the slope of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
          delta(nd.array): the y-intercept of the polynomial fit. Should be of shape (Num_channel,Num_gain) for our data
    """
    
    num_channels, num_gain = mean.shape[2], mean.shape[3]
    gain = np.zeros((num_channels, num_gain))
    delta = np.zeros((num_channels, num_gain))
    
    for i in range(num_channels):
        for j in range(num_gain):
            mask = mean[:, :, i, j] < th
            p = np.polyfit(mean[:, :, i, j][mask].flatten(), var[:, :, i, j][mask].flatten(), 1)
            gain[i, j] = p[0]
            delta[i, j] = p[1]
    
    return gain, delta

def fit_linear_polynom_to_read_noise(delta, gain):
    """
    finds the polyfit between mean and variance which you calculate in the previous functions, 
    mean and var.
    
    sigma(np.ndarray): the total read noise filtered into rgb values - #(Num_Channel,Num_gain)
    gain(np.ndarray): the estimated camera gain filtered into rgb values - #(Num_Channel,Num_gain)
    
    output:
          sigma_read(np.ndarray): the slope of the linear fit - #(Num_Channel)
          sigma_ADC(np.ndarray): the y-intercept of the linear fit - #(Num_Channel)
    """
    
    num_channels = delta.shape[0]
    sigma_read = np.zeros(num_channels)
    sigma_ADC = np.zeros(num_channels)
    
    for i in range(num_channels):
        p = np.polyfit(gain[i, :], delta[i, :], 1)
        sigma_read[i] = p[0]
        sigma_ADC[i] = p[1]
    
    return sigma_read, sigma_ADC
    
    
def calc_SNR_for_specific_gain(mean,var):

    """
    Calculate the SNR (mean / stddev) vs. the mean pixel intensity for a specific gain setting. You will need to bin the mean values into the range [0,255] so that you can compute SNR for a discrete set of values. 
    
    mean(np.ndarray): the mean of the img filtered into rgb values - #(M, N, Num_gain)
    var(np.ndarray): the variance of the img filtered into rgb values - #(M, N, Num_gain)
    
    output:
          SNR(np.ndarray): the computed SNR vs. mean of the captured image dataset - #(255, Num_gain)
    """
    
    num_bins = 255
    SNR = np.zeros(num_bins)
    
    mean_flat = mean.flatten()
    var_flat = var.flatten()
    
    for b in range(num_bins):
        mask = (mean_flat >= b) & (mean_flat < b + 1)
        if np.any(mask):
            SNR[b] = np.mean(mean_flat[mask]) / np.sqrt(np.mean(var_flat[mask]))
    
    return SNR