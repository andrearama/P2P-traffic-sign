#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:32:49 2018

@author: andrea
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, 'options')
sys.path.insert(0, 'util')
from test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import copy
import util as util1
from auxiliary import get_coordinates, brightening, increase_brighness, increase_saturation



############################################################################################################
def is_the_same(original, target):
    same_type = False        #In case target doesn't exist
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

def get_neir_images(original_frame_number,max_jump = 3, window_frame = 2):
    folder_path = "datasets/traffic_signs/snap_collection/"
    
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

        

# Add to average only the neighbours:
def get_smoothed_img(original_frame_number):
    
    img_serie = get_neir_images(original_frame_number, window_frame =1)
    v = []
    if len(img_serie) ==3:
        v.append(img_serie[0][0])
#        v.append(img_serie[1][0]) #averaging on the closest
#        v.append(img_serie[2][0]) #averaging on the closest
        
##        v.append(get_rgb_img(img_serie[0][0],img_serie[0][0]) )
##        v.append(get_rgb_img(img_serie[1][0],img_serie[1][0]) )
##        v.append(get_rgb_img(img_serie[2][0],img_serie[2][0]) )
        
    for img,nmber  in img_serie:
    
        if nmber == original_frame_number:
            img_o = img 
            v.append(img)
        else:
##            img2 = get_rgb_img(img_o,img)
##            v.append(img2) ### Average on close imgs histogrammed
            v.append(img) ### Average on close imgs unmodified
            pass
    img11 =np.mean(v, axis =0)
    img11 = img11.astype(np.uint8)
    return img11     
##########################################################################################################

def paste_image(l_img, s_img, x_offset,y_offset):
    """
    Paste a png image s_img on l_img removing the backround
    """ 
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])    
    return l_img

def paste_on_img(x_s, x_e, y_s, y_e, original_img, stamp):
    """
    Puts the patch on the original frame
    """
    if x_e>original_img.shape[1]:
        x_e = original_img.shape[1]
        print("TROPPO")
    if y_e>original_img.shape[0]:
        y_e = original_img.shape[0]
        print("TROPPO")
    w = x_e - x_s
    h = y_e - y_s

    stamp = cv2.resize(stamp,(w, h), interpolation = cv2.INTER_AREA)
    original_img[y_s:y_e, x_s:x_e] = stamp
    result = original_img
    return result


def save_images( visuals,namen,MC4BS, aspect_ratio=1.0, width=256):
    """
    Save the output of the GAN in the 'create' folder
    """
    for label, im_data in visuals.items():
        im = util1.tensor2im(im_data)
        save_path = "datasets/traffic_signs/create/"+namen #in our case its BB.jpg
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        if "fake" in label:
            global number 
            util1.save_image(im, save_path)
            if second_cycle == False:

                img_p = image_tot_mod
                img_o = original_img
                pathpath = "datasets/traffic_signs/snap_collection/"+str(number)+".jpg"
#                im = increase_brighness(im,pathpath,name,img_p,img_o,coordinates,MC4BS)
#                im = increase_saturation(im,pathpath,name,img_p,img_o,coordinates,MC4BS)
                util1.save_image(im, pathpath)




def get_focused_squared_data(img,xmin,ymin,xmax,ymax, name, original_img,
                     display_result = False,particular_case = False) :
    """
    Outputs the final frame confrontation
    """
    original_img1 = copy.deepcopy(original_img)
    p = 0.0001
    size = 256
    
    width = xmax - xmin
    height = ymax - ymin
    xcenter = int((xmax+xmin)/2)
    ycenter = int((ymax+ymin)/2)    
    
    if width>height: 
        y_s = ycenter-int(width/2)-int(width*p)
        y_e = ycenter+int(width/2)+int(width*p)
        x_s = xmin-int(width*p)
        x_e = xmax+int(width*p)
    else:
        y_s = ymin-int(height*p)
        y_e = ymax+int(height*p)
        x_s = xcenter-int(height/2)-int(height*p)
        x_e = xcenter+int(height/2)+int(height*p)
    if x_s <0:
        x_s = 0
    if y_s <0:
        y_s =0
    ### This to enlarge image 
    x_e = xmax
    y_e = ymax
    x_s = xmin
    y_s = ymin
    ###        
    squared_data = img[y_s:y_e,x_s:x_e] 
    original_data = original_img[y_s:y_e,x_s:x_e] 
    MC4BS = (y_s,y_e,x_s,x_e)
    

    squared_data = cv2.resize(squared_data,(size, size), interpolation = cv2.INTER_CUBIC)
    original_data = cv2.resize(original_data,(size, size), interpolation = cv2.INTER_CUBIC)
    vis = np.concatenate((original_data,squared_data), axis=1)
    cv2.imwrite("datasets/traffic_signs/test/AB.jpg",vis)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    
#        print("LENGTH: ",len(dataset))
    
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        save_images( visuals,"BB.jpg",MC4BS)
        if second_cycle:
#                brightening() # If so create folder CHECK close to near
            stamp = get_smoothed_img(number)
            result = paste_on_img(x_s, x_e, y_s, y_e, original_img1, stamp)
            final = np.concatenate((original_img, result), axis=1)
            visualize = cv2.resize(final,None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
        
            # Show the complete frames (final: original size, visualize: halfed one)
            cv2.imshow('image',visualize)
            cv2.imwrite('datasets/risu/'+str(number)+'.jpg', visualize)
            cv2.imwrite('datasets/ris_frame/'+name+'.jpg',result)
            cv2.waitKey(1)            

    if second_cycle:        
        return final
    else:
        return []

    
    
def create_comparison(name, ls_image, uturn_image, ra_image,noentry_image, 
                    display_result = False) :
    """
    From the initial image, using the above functions, it outputs confrontation
    """

    name_img = path_for_1 + '1/JPEGImages/'+name+'.jpg'
    img = cv2.imread(name_img)

    global original_img
    original_img = copy.deepcopy(img)
    
    #Assumption: only one sign to modify. Check older version for multilpe

    name_obj = coordinates[name]["type"]
    print(name_obj)
    xmin = coordinates[name]["xmin"]
    ymin = coordinates[name]["ymin"]
    xmax = coordinates[name]["xmax"]
    ymax = coordinates[name]["ymax"]


    if name_obj in ['ls05','ps22', 'fs17','gs01']:
        if second_cycle == False:
            global kt_list
            kt_list.append((number,name_obj))

                
        ls_image = cv2.resize(ls_image,(xmax-xmin,ymax-ymin), interpolation = cv2.INTER_AREA)
        img = paste_image(img, ls_image, xmin,ymin)
        global image_tot_mod #@@@@@@@@@@@@@@@
        image_tot_mod = img  #@@@@@@@@@@@@@@@
        fifi = get_focused_squared_data(img,xmin,ymin,xmax,ymax, name, original_img)
    else:
      #  cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)    
      pass
  
    if display_result:
       
        cv2.imshow('image',fifi)
        cv2.waitKey(5)
            
    return 


if __name__ == '__main__':
    
    #Load images:
    ls_image = cv2.imread("plain_images/ls05.png", -1)
    uturn_image = cv2.imread("plain_images/upleft.png", -1) 
    ra_image = cv2.imread("plain_images/rightarrow.png", -1)     
    noentry_image = cv2.imread("plain_images/noentry.png", -1) 
   # noturn_image = cv2.imread("plain_images/ntr.png", -1) 
    noturn_image = cv2.imread("plain_images/ls05.png", -1)     

    #GAN settings
    opt = TestOptions().parse()
    opt.nThreads = 1   
    opt.batchSize = 1  
    opt.serial_batches = True  
    opt.no_flip = True  
    opt.display_id = -1  
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    path_for_1 = opt.original_dataset_directory 
    folder_path = "datasets/traffic_signs/test/"

    
    #Prep gloabal lists
    coordinates = get_coordinates()
    kt_list = []
    
    second_cycle= False
    F = open(path_for_1+'1/ImageSets/1.txt','r')
    lines = F.readlines()
    for name in lines :
        number = int(name[3:])
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",number)

        if(number>1865 and number<7256) or (number>146 and number<1043) or (number>1960 and number<6800) or (number>7421 and number<8833) :
#        if (number>146 and number<1043) or (number>1960 and number<6800) or (number>7421 and number<8833) :        
            pass
        else:                
            name = name[:-1]
            img = create_comparison(name,noturn_image, uturn_image,ra_image,noentry_image,
                             display_result = False) 

                    
    second_cycle = True 
    F = open(path_for_1+'1/ImageSets/1.txt','r')
    lines = F.readlines()
    for name in lines :
        number = int(name[3:])
        print("#########################################",number)

        if(number>1865 and number<7256) or (number>146 and number<1043) or(number>1960 and number<6800) or (number>7421 and number<8833) :
#        if (number>146 and number<1043) or (number>1960 and number<6800) or (number>7421 and number<8833) :        
            pass
        else:                
            name = name[:-1]
            create_comparison(name,noturn_image, uturn_image,ra_image,noentry_image,
                             display_result = False) 



