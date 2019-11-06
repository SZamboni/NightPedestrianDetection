import cv2
import numpy as np
from skimage import exposure as ex
from skimage import data
from PIL import Image
import skfuzzy as fuzz
import math
import timeit
import time

'''
Histogram equalization with colour YCR_CB and histogram equalization only on Y

@img: the image to modify

@return: the image with the histogram equalized
'''
def hisEqulColor_YCRCB(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    result = cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2RGB)
    return result

'''
Histogram equalization with colour YCR_CB and histogram equalization only on Y

@img: the image to modify

@return: the image with the histogram equalized
'''
def hisEqulColor_HSV(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    result = cv2.cvtColor(ycrcb,cv2.COLOR_HSV2RGB)
    return result

# Histrogram equalization from 
# https://github.com/AndyHuang1995/Image-Contrast-Enhancement/blob/master/he.py
'''
Histogram equalization equalizing every colour

@img: the image to modify

@return: the image with the histogram equalized
'''
def hisEqulColor_RGB(img):
    outImg = np.zeros((img.shape[0],img.shape[1],3))
    for channel in range(img.shape[2]):
        outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

'''
Gamma correction

@imgage: the image to modify
@gamma: the gamma value, 1.0 does nothing

@return: the image with the gamma corrected
'''
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

# Adaptive gamma correction based on the reference.
# Reference:
#   S. Huang, F. Cheng and Y. Chiu, "Efficient Contrast Enhancement Using Adaptive Gamma Correction With
#   Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041,
#   March 2013. doi: 10.1109/TIP.2012.2226047
# Revised from https://github.com/mss3331/AGCWD/blob/master/AGCWD.m

#from https://github.com/qyou/AGCWD/blob/master/agcwd.py

import numpy as np
import cv2

'''
Adaptive gamma correction with Weighting Distribution

@image: the image to modify
@w: the weight distribution

@return: the image with the gamma corrected
'''
def agcwd(image, w=0.5):
    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image


def extract_value_channel(color_image):
    color_image = color_image.astype(np.float32) / 255.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return np.uint8(v * 255)


def get_pdf(gray_image):
    height, width = gray_image.shape
    pixel_count = height * width
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist / pixel_count


def set_value_channel(color_image, value_channel):
    value_channel = value_channel.astype(np.float32) / 255
    color_image = color_image.astype(np.float32) / 255.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image[:, :, 2] = value_channel
    color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
    return color_image


# Then we sould have from https://github.com/AndyHuang1995/Image-Contrast-Enhancement/blob/master/ying.py


# from https://www.programcreek.com/python/example/89353/cv2.createCLAHE,
# CLAHE (Contrast-limited adaptive histogram equalization) 
'''
Function that apply CLAHE (Contrast-limited adaptive histogram equalization)
to every channel of the image

@imgage: the image to modify

@return: the image with the histrogram corrected

'''
def applyCLAHE(image, clip_limit=3):
    
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

# RETINEX from https://github.com/dongb5/Retinex/blob/master/

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):    

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img    

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       

    return img_msrcr

def automatedMSRCR(img, sigma_list):

    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)
        
    return img_retinex

'''
Function that apply MSRCP (Multi Scale Retinex

@img: the image to modify
@sigma_list: the list of the sigma, by default [15,80,250]

@return: the image with the histrogram corrected

'''
def MSRCP(img, sigma_list = [15,80,250], low_clip = 0.01, high_clip = 0.99):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]    

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)
    
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp

