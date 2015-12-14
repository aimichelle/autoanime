import numpy as np
import cv2
import math
from scipy import ndimage
from matplotlib import pyplot as plt
import autoanime
import os
from scipy import signal
from PIL import Image, ImageDraw
from scipy.cluster.vq import kmeans,vq
import colorsys

def hair_color(shapes, fname,mask):

    img = cv2.imread(fname)

    mask = cv2.cvtColor(mask,  cv2.COLOR_GRAY2BGR);
    cv2.bitwise_and(img, mask, img);
    mask = mask[shapes.part(20).y-(shapes.part(33).y - shapes.part(27).y)*1.65: shapes.part(20).y - (shapes.part(33).y - shapes.part(27).y)/4, shapes.part(0).x:shapes.part(16).x]
    img = img[shapes.part(20).y-(shapes.part(33).y - shapes.part(27).y)*1.65: shapes.part(20).y - (shapes.part(33).y - shapes.part(27).y)/4, shapes.part(0).x:shapes.part(16).x]
    mask = cv2.cvtColor(mask,  cv2.COLOR_BGR2GRAY);

    num_masked = len(mask.ravel()[np.flatnonzero(mask)])
   
    cv2.imwrite("cropped_hair.jpg",img)
    i = Image.open("cropped_hair.jpg")
    h = i.histogram()

    r = h[1:256]
    g = h[256+1:256*2]
    b = h[256*2+1: 256*3]
   

    r = sum( i*w for i, w in enumerate(r) ) / (sum(r)) #- num_masked)
    g = sum( i*w for i, w in enumerate(g) ) / (sum(g)) #- num_masked)
    b =  sum( i*w for i, w in enumerate(b) ) / (sum(b)) #- num_masked)


    h,l,s = colorsys.rgb_to_hls(r/255.,g/255., b/255.)
    l = l + .05
    r,g,b = colorsys.hls_to_rgb(h,l,s)
    print (r*255,g*255,b*255)
    return (r*255,g*255,b*255)

def draw_long_hair(shapes,hair_file,im, color):
    face_width = shapes.part(0).x - shapes.part(16).x
    ratio = -face_width/400.

    if "ncc" in hair_file:
        hair_idx = hair_file[-7]
        hair_file = "hair/"+hair_idx + "-l.png"

    hair_im = Image.open(hair_file)
    #color hair
    print color 
    if color[0] < 55 and color[1] < 55 and color[2] < 55: #black hair
        #black hair, greyscale it.
        hair_im = hair_im.convert('L')
        print 'hair turned to black'
    else:
        color_hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        hue = color_hsv[0] * 360
        hair_im = autoanime.colorize(hair_im, hue)
        print 'colorized hair to hue ', hue

    pt_0 = ((hair_im.size[0]/2. - 200)*ratio, 410*ratio) 
    hair_im = hair_im.resize((int(hair_im.size[0]*ratio), int(hair_im.size[1]*ratio)),resample=Image.BICUBIC)
    shift_x = int(shapes.part(0).x - pt_0[0])
    shift_y = int(shapes.part(0).y - pt_0[1])
    im.paste(hair_im, box=(shift_x,shift_y), mask=hair_im)
    return im


def draw_hair(shapes,hair_file,im, angle, color):
    # resize hairfile
    face_width = shapes.part(0).x - shapes.part(16).x
    ratio = -face_width/400.

    if "ncc" in hair_file:
        hair_idx = hair_file[-7]
        hair_file = "hair/"+hair_idx + "-s-f.png"

    hair_im = Image.open(hair_file)
    #change color
    #default RGB = 128,83,48
    print color 
    if color[0] < 55 and color[1] < 55 and color[2] < 55: #black hair
        #black hair, greyscale it.
        hair_im = hair_im.convert('LA')
        print 'hair turned to black'
    else:
        color_hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        print 'hair hsv is ', color_hsv
        hue = color_hsv[0] * 360
        hair_im = autoanime.colorize(hair_im, hue)
        print 'colorized hair to hue ', hue


    pt_0 = ((hair_im.size[0]/2. - 200)*ratio, 410*ratio) 
    hair_im = hair_im.resize((int(hair_im.size[0]*ratio), int(hair_im.size[1]*ratio)),resample=Image.BICUBIC)
    
    #then rotate it
    hair_im = hair_im.rotate(angle, resample=Image.BICUBIC, expand = 1)
    print "we rotated the hair by ", angle

    shift_x = int(shapes.part(0).x - pt_0[0])
    shift_y = int(shapes.part(0).y - pt_0[1])
    im.paste(hair_im, box=(shift_x,shift_y), mask=hair_im)
    return im

def match_hair(mask,shapes,long_hair,gender): # face width, img width/2 shapes.part(0).x is to left by facewidth/2
    # resize mask so that 1 to 17 is 400, 410 pixels down  dont check lng hairs
    print "matching hair..."
    face_width = shapes.part(0).x - shapes.part(16).x
    shift_x = mask.shape[1]/2 - shapes.part(0).x + face_width/2
    rows,cols = mask.shape
    M = np.float32([[1,0,shift_x],[0,1,0]])
    mask = cv2.warpAffine(mask,M,(cols,rows))
    ratio = -400./face_width
    
    mask = cv2.resize(mask, (int(mask.shape[1]*ratio), int(mask.shape[0]*ratio)))
    rows,cols = mask.shape
    shift_y = 410 - shapes.part(0).y*ratio 
    M = np.float32([[1,0,0],[0,1,shift_y]])
    mask = cv2.warpAffine(mask,M,(cols,rows))
    mask = cv2.resize(mask, (int(mask.shape[1]/3), int(mask.shape[0]/3)))
    mask_normed = (mask - mask.mean()) / mask.std()
    # cv2.imwrite("resized_mask.jpg",mask)
    scores = []
    fnames = []
    for filename in os.listdir('hair'):
        if filename == '.DS_Store':
            continue

        if (long_hair and "ncc" not in filename) or (not long_hair and "s" not in filename): 
            continue               
        im = cv2.imread('hair/'+filename,0)
        im[im == 255] = 0
        im[im > 0] = 255
        im = cv2.resize(im, (int(im.shape[1]/3), int(im.shape[0]/3)))
        fnames.append('hair/'+filename)

        anime_normed = (im - im.mean()) / im.std()

        ncc = signal.correlate2d(mask_normed, anime_normed, mode='same')
        if (gender == 'f' and 'f' in filename) or (gender == 'm' and 'm' in filename):
            scores.append(ncc.max()*1.25)
        else:
            scores.append(ncc.max())

    #     print scores
    #     print filename
    print scores
    print fnames
    print fnames[np.argmax(scores)]
    return fnames[np.argmax(scores)]
        # size_x - offset

        

        #     padded_mask[0:size_y,offset:size_x] = mask[:size_y,:size_x]
        # else:
        #     padded_mask[0:size_y,offset:size_x] = mask[:size_y,:size_x-offset]

        # center shapes

        # cv2.imwrite("resized_mask.jpg",im)

def get_hair_mask(fname):
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get bounding box
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     # cv2.rectangle(img,(int(x + w/2 - float(5)/3/2*w), int(y - h/2)),
    #     #                   (int(x + w/2 + float(5)/3/2*w), int(y+3*h/2)),(255,0,0),2)
    #     cv2.rectangle(img,(x, y), (x+w, y+h),(255,0,0),2)
    # bound_box = (x,y,w,h)

    shapes = autoanime.predict_shape(fname)
    # cv2.imwrite("img_0.png",img)

    # Frequential mask
    img = cv2.imread(fname,0)

    filtered_img = ndimage.filters.gaussian_filter(img,60) #20 #40
    mask = img - filtered_img 

    threshold = 170 # 100
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255
    # cv2.imwrite("img_4.png",mask)

    # Crop to get hair
    img = cv2.imread(fname)
    img = img[shapes.part(20).y-(shapes.part(33).y - shapes.part(27).y)*1.5: shapes.part(20).y - (shapes.part(33).y - shapes.part(27).y)/4, shapes.part(0).x:shapes.part(16).x]
    cv2.imwrite("img_4.png",img)
    r,g,b = cv2.split(img)
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)

    img = cv2.imread(fname)
    r,g,b = cv2.split(img)
    r[r > mean_r + std_r] = 0
    r[r < mean_r - std_r] = 0
    r[r > 0] = 255
    b[b > mean_b + std_b] = 0
    b[b < mean_b - std_b] = 0
    b[b > 0] = 255
    g[g > mean_g + std_g] = 0
    g[g < mean_g - std_g] = 0
    g[g > 0] = 255

    # Mask with frequency analysis and hair color
    color_mask = np.zeros((len(r), len(r[0])))
    for i in range(len(r)):
        for j in range(len(r[0])):
            if r[i][j] == 255 and b[i][j] == 255 and g[i][j] == 255 and mask[i][j] == 255:
                color_mask[i][j] = 255


    color_mask[shapes.part(38).y:shapes.part(56).y,shapes.part(36).x:shapes.part(45).x] = 0
    cv2.imwrite("mask.jpg",color_mask)

    if not long_hair(color_mask, shapes):
        color_mask[shapes.part(3).y:] = 0

    color_mask = np.uint8(color_mask)

    # Try to fill in holes
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(color_mask,kernel,iterations = 2)
    erosion = cv2.erode(dilation,kernel,iterations=2)
    
    return erosion


def get_hair_angle(shape):
    """returns how much we need to rotate the hair based on eye orientation."""
    rise = abs(shape.part(36).y - shape.part(45).y)
    run = abs(shape.part(36).x - shape.part(45).x)
    angle = math.degrees(math.atan(float(rise/run)))
    if shape.part(36).y > shape.part(45).y: #the left eye is higher, rotate clockwise
        return 360 - angle
    else:
        return angle

def long_hair(mask,shapes):

    mask2 = mask[shapes.part(3).y:shapes.part(5).y].copy()
    cv2.imwrite("hair_crop.jpg",mask)
    mask2[mask2 > 0] = 1
    mask_array = np.asarray(mask2)
    mask_array.sum()/float(mask2.size)
    # print mask_array.sum()/float(mask.size)
    print "Long hair fraction: " + str(mask_array.sum()/float(mask2.size))
    print "Long hair? " + str(mask_array.sum()/float(mask2.size) > .1)
    return mask_array.sum()/float(mask2.size) > .1