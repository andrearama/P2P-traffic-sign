#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:46:47 2018

@author: andrea
"""
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy


def is_the_same(original, target):
    #find original name
    global kt_list
    for num,nam in kt_list:
        if num == original:
            nam_o = nam
            break #Problem if more than one in the original frame
            
    for num,nam in kt_list:
        if num == target:
            same_type = (nam == nam_o)
            break
    return same_type

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape)

def get_rgb_img(img,img2,plot_set = False):
    img3 = copy.deepcopy(img)
    
    for i in range(0,3):
    
        source = img[:,:,i]
        template = img2[:,:,i]
        matched = hist_match(source, template)    
        img3[:,:,i] = matched
        
        if plot_set:
            res = np.hstack((img,img2,img3)) 
            cv2.imshow('res.png',res)
            
    return img3

def get_neir_images(original_frame_number,max_jump = 4, window_frame = 2):
    folder_path = "/home/andrea/Documents/NCTU/pytorch-CycleGAN-and-pix2pix/datasets/facades/snap_collection/"
    
    img_serie = [(cv2.imread(folder_path+str(original_frame_number)+'.jpg'),original_frame_number)]
    
    jump_count_fw_total = 0
    jump_count_fw = 0    
    frame_number = original_frame_number 
    while(True):
        frame_number = frame_number + 1    

        condition = is_the_same(original_frame_number, frame_number) 
                
        img = cv2.imread(folder_path+str(frame_number)+'.jpg')
        if None is img or condition == False :
            jump_count_fw = jump_count_fw +1
            jump_count_fw_total = jump_count_fw_total +1
            if jump_count_fw > max_jump:
                break
        else:
            jump_count_fw = 0  
    
            if (frame_number-original_frame_number) > window_frame + jump_count_fw_total:
               break
            else:
                img_serie.append( (img,frame_number) )

        if (frame_number-original_frame_number) > window_frame + jump_count_fw_total:
           break        
    
    jump_count_bw_total = 0
    jump_count_bw = 0   
    frame_number = original_frame_number 
    while(True):
        frame_number = frame_number - 1    

        condition = is_the_same(original_frame_number, frame_number) 
        
        img = cv2.imread(folder_path+str(frame_number)+'.jpg')
        if None is img or condition == False :
            jump_count_bw = jump_count_bw +1
            jump_count_bw_total = jump_count_bw_total +1
            if jump_count_bw > max_jump:
                break
        else:
            jump_count_bw = 0  
    
            if original_frame_number-frame_number > window_frame + jump_count_bw_total :
               break  
            else:
               img_serie.append( (img,frame_number) )

        if original_frame_number-frame_number > window_frame + jump_count_bw_total :
           break  

    return img_serie        

#img = cv2.imread('/home/andrea/Documents/NCTU/pytorch-CycleGAN-and-pix2pix/datasets/facades/create/BB.jpg')

        


def get_smoothed_img(original_frame_number):
    
    img_serie = get_neir_images(original_frame_number)
    print(len(img_serie))
    v = []
    for img,nmber  in img_serie:
    
        if nmber == original_frame_number:
            img_o = img 
        else:
            img2 = get_rgb_img(img_o,img)
            v.append(img2)
    img11 =np.mean(v, axis =0)
    img11 = img11.astype(np.uint8)
    return img11        
#
##main:
#for original_frame_number in range(60,103):
#    i = get_smoothed_img(original_frame_number)
#    o = cv2.imread("/home/andrea/Documents/NCTU/pytorch-CycleGAN-and-pix2pix/datasets/facades/snap_collection/"+str(original_frame_number)+'.jpg')
#    cv2.imshow("",np.hstack((i,o)))
#    cv2.waitKey(500)