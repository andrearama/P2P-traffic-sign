#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:14:58 2018

@author: andrea
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import copy
from PIL import Image, ImageStat
import math

def extract_coordinates(name, coordinates, modd = False) :
    """
    
    """
    if modd:
        name_parse = '/media/andrea/MY PASSPORT/HERE/ALTIS_PC/XML/'+name+'.xml'
    else:
        name_parse = '1/Annotations/'+name+'.xml'
    
    tree = ET.parse(name_parse)
    root = tree.getroot()
    for i in range(2,len(root)):
        name_obj = root[i][0].text
        xmin =int( root[i][1][0].text )
        ymin =int( root[i][1][1].text ) 
        xmax =int( root[i][1][2].text )
        ymax =int( root[i][1][3].text )

#        xmin = xmin - int((xmax-xmin)/20) #da 20
#        ymin = ymin - int((ymax-ymin)/20) 
        xmin1 = xmin
        ymin1 = ymin
        xmin = xmin - int((xmax-xmin)/12) #da 20
        ymin = ymin - int((ymax-ymin)/12)
        xmax = xmax + int((xmax-xmin1)/12)
        ymax = ymax + int((ymax-ymin1)/12)
     #   print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        if ymax >1079: 
            ymax = 1079
        if xmax >1919: 
            xmax = 1919

        if xmin <1: 
            xmin = 1
        if ymin <1: 
            ymin = 1

        try:
            if coordinates[name]["type"] not in  ['ls05','ps22', 'fs17','gs01']:
               coordinates[name] = {"xmin":xmin, "ymin": ymin, "xmax":xmax, "ymax":ymax,"type":name_obj}
        except   KeyError:    
            coordinates[name] = {"xmin":xmin, "ymin": ymin, "xmax":xmax, "ymax":ymax,"type":name_obj}

    return coordinates            

def transform_number(nn):
    nn = str(nn)
    string  = "1_"+'0'*(5-len(nn))+nn
    return string
    

def average_coordinates(coordinates):
    new_coordinates = copy.deepcopy(coordinates)
    nn = 0
    breaker = 0
    
    while True:        
        name = transform_number(nn)      
        n_o = copy.deepcopy(nn)
        xmin_v =[]
        xmax_v =[]
        ymin_v =[]
        ymax_v =[]
        try:
            ori = coordinates[name]["type"]
            xmin_v.append( coordinates[name]["xmin"] )
            xmax_v.append( coordinates[name]["xmax"] )
            ymin_v.append( coordinates[name]["ymin"] )
            ymax_v.append( coordinates[name]["ymax"] )
    
            nn = n_o-1
            name = transform_number(nn)
            try:
                if coordinates[name]["type"] == ori:
                    xmin_v.append( coordinates[name]["xmin"] )
                    xmax_v.append( coordinates[name]["xmax"] )
                    ymin_v.append( coordinates[name]["ymin"] )
                    ymax_v.append( coordinates[name]["ymax"] )
            except KeyError:            
                pass
    
            nn = n_o+1
            name = transform_number(nn)
            try:
                if coordinates[name]["type"] == ori:
                    xmin_v.append( coordinates[name]["xmin"] )
                    xmax_v.append( coordinates[name]["xmax"] )
                    ymin_v.append( coordinates[name]["ymin"] )
                    ymax_v.append( coordinates[name]["ymax"] )
            except KeyError:            
                pass

            nn = n_o+2
            name = transform_number(nn)
            try:
                if coordinates[name]["type"] == ori:
                    xmin_v.append( coordinates[name]["xmin"] )
                    xmax_v.append( coordinates[name]["xmax"] )
                    ymin_v.append( coordinates[name]["ymin"] )
                    ymax_v.append( coordinates[name]["ymax"] )
            except KeyError:            
                pass
            
            nn = n_o-2
            name = transform_number(nn)
            try:
                if coordinates[name]["type"] == ori:
                    xmin_v.append( coordinates[name]["xmin"] )
                    xmax_v.append( coordinates[name]["xmax"] )
                    ymin_v.append( coordinates[name]["ymin"] )
                    ymax_v.append( coordinates[name]["ymax"] )
            except KeyError:            
                pass            

            nn = n_o+1
            name = transform_number(n_o)
            new_coordinates[name]["xmin"] = int(np.mean(xmin_v) ) 
            new_coordinates[name]["xmax"] = int(np.mean(xmax_v) )
            new_coordinates[name]["ymin"] = int(np.mean(ymin_v) )
            new_coordinates[name]["ymax"] = int(np.mean(ymax_v) )
            print(name,"------>", len(xmin_v))
        except KeyError:    
            pass

        #loop breaker:            
        if len(xmin_v) >0 : 
            breaker = 0
        else:
            breaker = breaker +1
        if breaker >50000 :
            print("break!",breaker)
            break 

    return new_coordinates
                
def get_coordinates():
    coordinates = {} 

    F = open('1/ImageSets/1.txt','r')
    lines = F.readlines()
    for name in lines :
        number = int(name[3:])
        if (number>1960 and number<6800)  or (number>7421 and number<8833) :
            pass
        else:                
            name = name[:-1]
            coordinates = extract_coordinates(name,coordinates) 
    #IF WANT TO SMOOTH:            
    #coordinates = average_coordinates(coordinates)
    return coordinates
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def get_coordinates_mod(lines):
    coordinates = {} 

    for number, name in lines :
       
        #name = name[:-1]
        coordinates = extract_coordinates(name,coordinates, modd = False) 
#IF WANT TO SMOOTH:            
    coordinates = average_coordinates(coordinates)
    return coordinates
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

############################################################

def increase_brightness(img, value):
    value = int(value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] = v[v >= lim] +value
        
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def compute_brightness_diff(img_o,img_c):
    hsv_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)
    h_o, s_o, v_o = cv2.split(hsv_o)
    h_c, s_c, v_c = cv2.split(hsv_c)
    
    diff = np.mean(v_o-v_c) #/5
    return diff

def brightening():
    intera = cv2.imread("datasets/traffic_signs/test/AB.jpg")
    img_original_real = intera[:,0:256]
    img_original_paste = cv2.imread("datasets/sbatta/AB.jpg")
    img_modified_fake = intera[:,256:]
    img_modified_gan =  cv2.imread("datasets/traffic_signs/create/BB.jpg")
    
    d1 = compute_brightness_diff(img_original_real, img_original_paste)
    d2 = compute_brightness_diff(img_modified_fake, img_modified_gan)
    d3 = compute_brightness_diff(img_original_real, img_modified_gan)

    result = increase_brightness(img_modified_gan, value=d3)
    print(d3)

    cv2.imwrite("datasets/traffic_signs/create/BB.jpg",result)
    
    #To check that it's right!    
    final = np.concatenate((img_modified_gan, result), axis=1)
    cv2.imwrite("datasets/CHECK/BB.jpg", final)

    return 

###############################################################
    
def increase_brightness_tw(img, value,im_o,im_p,coordinates,name,MC4BS):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    y_s,y_e,x_s,x_e = MC4BS
    
    bob = im_o[y_s:y_e,x_s:x_e] 
    alice = im_p[y_s:y_e,x_s:x_e] 
    sputo = copy.deepcopy(bob) #Proofa
    for y in range(bob.shape[0]):
        for x in range(bob.shape[1]):
            if (alice[y,x] == bob[y,x]).all() :
                sputo[y,x] = bob[y,x]*0
            else:
                sputo[y,x] = bob[y,x]*0 +1

    sputo = cv2.resize(sputo, (256, 256), interpolation = cv2.INTER_CUBIC)
    bob   = cv2.resize(bob, (256, 256), interpolation = cv2.INTER_CUBIC)
    

        
    value = int(value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] = v[v >= lim] +value
        
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # Restoring the original background:
    img = img*sputo +bob*(1-sputo)
    
#    cv2.imshow("bo",img)
#    cv2.waitKey(1) 


    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#perceived brightness:
def brightness3( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def brightness_around(img_p,img_o,name, coordinates):
    xmin = coordinates[name]["xmin"] 
    ymin = coordinates[name]["ymin"]
    xmax = coordinates[name]["xmax"]
    ymax = coordinates[name]["ymax"]    

    xmin -= int( (xmax-xmin)/2.7 )
    ymin -= int( (ymax-ymin)/2.7 )
    xmax += int( (xmax-xmin)/2.7 )
    ymax += int( (ymax-ymin)/2.7 )
        
    if xmin <1: xmin = 1
    if ymin <1: ymin = 1
    if xmax> 1919: xmax=1919
    if ymax >539: ymax = 539
    
    v = []
    for y in range(ymin,ymax):
        for x in range(xmin,xmax):
            if (img_p[y][x] == img_o[y][x]).all():
              b = img_o[y][x][0]
              g = img_o[y][x][1]
              r = img_o[y][x][2]
              v.append( math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) )  
            
    return np.mean(v)



def increase_brighness(im,pathpath,name,img_p,img_o,coordinates,MC4BS):
    br_target = brightness_around(img_p,img_o,name, coordinates)
    br_actual = brightness3( pathpath )
    delta_brigntess = br_target- br_actual
#    if delta_brigntess>50: delta_brigntess = delta_brigntess*1.6
#    if delta_brigntess>35: delta_brigntess = delta_brigntess*1.2
    img_target = increase_brightness_tw(im, delta_brigntess,img_o,img_p,coordinates,name,MC4BS)
    print("Brighness increase:",delta_brigntess)
    return img_target


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def compute_saturation(pathpath):
    img_o = cv2.imread(pathpath)
    hsv_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
    h_o, s_o, v_o = cv2.split(hsv_o)
    
    return np.mean(s_o)

def saturation_around(img_p,img_o,name, coordinates):
    hsv_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
    h_o, s_o, v_o = cv2.split(hsv_o)
      
    xmin = coordinates[name]["xmin"] 
    ymin = coordinates[name]["ymin"]
    xmax = coordinates[name]["xmax"]
    ymax = coordinates[name]["ymax"]    

    xmin -= int( (xmax-xmin)/2.7 )
    ymin -= int( (ymax-ymin)/2.7 )
    xmax += int( (xmax-xmin)/2.7 )
    ymax += int( (ymax-ymin)/2.7 )
        
    if xmin <1: xmin = 1
    if ymin <1: ymin = 1
    if xmax> 1919: xmax=1919
    if ymax >539: ymax = 539
    
    v = []
    for y in range(ymin,ymax):
        for x in range(xmin,xmax):
            if (img_p[y][x] == img_o[y][x]).all():
              v.append( s_o[y][x] )  

    return np.mean(v)

def increase_saturation_tw(img, value,im_o,im_p,coordinates,name,MC4BS):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    y_s,y_e,x_s,x_e = MC4BS
    
    bob = im_o[y_s:y_e,x_s:x_e] 
    alice = im_p[y_s:y_e,x_s:x_e] 
    sputo = copy.deepcopy(bob) #Proofa
    for y in range(bob.shape[0]):
        for x in range(bob.shape[1]):
            if (alice[y,x] == bob[y,x]).all() :
                sputo[y,x] = bob[y,x]*0
            else:
                sputo[y,x] = bob[y,x]*0 +1

    sputo = cv2.resize(sputo, (256, 256), interpolation = cv2.INTER_CUBIC)
    bob   = cv2.resize(bob, (256, 256), interpolation = cv2.INTER_CUBIC)
    

        
    value = int(value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, v, s = cv2.split(hsv)
    if value >0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] = v[v >= lim] +value
        
    final_hsv = cv2.merge((h, v, s))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # Restoring the original background:
    img = img*sputo +bob*(1-sputo)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def increase_saturation(im,pathpath,name,img_p,img_o,coordinates,MC4BS):
    br_target = saturation_around(img_p,img_o,name, coordinates)
    br_actual = compute_saturation( pathpath )
    delta_saturation = br_target- br_actual
#    if delta_saturation>40: delta_saturation = delta_saturation*1.1
    img_target = increase_saturation_tw(im, delta_saturation,img_o,img_p,coordinates,name,MC4BS)
    print("Saturation increase:",delta_saturation)
    return img_target


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    


#im_file = "/home/andrea/Documents/NCTU/pytorch-CycleGAN-and-pix2pix/1/JPEGImages/1_00002.jpg"
#print(brightness1(im_file), brightness2(im_file), brightness3(im_file), brightness4(im_file), brightness5(im_file))
#def change_brighness:

    
#img_o = cv2.imread("/home/andrea/Music/original.png")
#img_p = cv2.imread("/home/andrea/Music/mod.png")        
#print(brightness_around(img_p,img_o,name, coordinates))
#
#img_p2 = cv2.imread("/home/andrea/Music/mod2.png") 
#img_p3 = cv2.imread("/home/andrea/Music/mod3.png") 
#print(brightness_around(img_p2,img_p3,name, coordinates))
#
#
#name = "1_00125" #------------------------------------<<<<<
#xmin = coordinates[name]["xmin"] 
#ymin = coordinates[name]["ymin"]
#xmax = coordinates[name]["xmax"]
#ymax = coordinates[name]["ymax"]   
#
#xmin -= int( (xmax-xmin)/3 )
#ymin -= int( (ymax-ymin)/3 )
#xmax += int( (xmax-xmin)/3 )
#ymax += int( (ymax-ymin)/3 )
#
#cv2.rectangle(img_p2,(xmin,ymin),(xmax,ymax),(0,255,0),3) 
#cv2.imshow("",img_p2)
#



def create_video() :   
    s = (540,1920,3)
    height , width , layers =  s
    
    video = cv2.VideoWriter('video_presentation.avi',cv2.VideoWriter_fourcc(*"XVID"), 8,(width,height))
    
    path = "datasets/risu/"
    F = open('1/ImageSets/1.txt','r')
    lines = F.readlines()
    for name in lines :
        number = int(name[3:])
        
        img = cv2.imread(path+str(number)+".jpg")
        
        if img is not None:
            if img.shape == s:  
                video.write(img)
                print("#"*int(number*60/9000))
            else:
                print(img.shape)
    
    
    cv2.destroyAllWindows()
    video.release()
    
