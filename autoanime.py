#!/usr/bin/python

import cv2
import sys
import os
import math
import dlib
import glob
from skimage import io
from PIL import Image, ImageDraw
import numpy as np
from scipy.cluster.vq import kmeans,vq
from pylab import imread,imshow,show


MODEL = "shape_predictor_68_face_landmarks.dat"

def autoanime(fname):
    shape = predict_shape(fname)

    orig_im = Image.open(fname)
    new_im = Image.new("RGB", (orig_im.size[0], orig_im.size[1]), color=(255,255,255))

    skin_color = quantize_skin(fname,shape)

    new_im = color_skin(new_im, shape, skin_color)
    
    

    # Draw outline
    new_im = draw_lineart(new_im, shape)

    # Save image
    new_im.save("test.png", "PNG")

def color_skin(im, shape, colors):
    # Find darkest and lightest color using luminance
    if (0.2126*colors[0][0] + 0.7152*colors[0][1] + 0.0722*colors[0][2]) > (0.2126*colors[1][0] + 0.7152*colors[1][1] + 0.0722*colors[1][2]):
        base_idx = 0
    else:
        base_idx = 1
    draw = ImageDraw.Draw(im)

    # face
    draw.polygon([(shape.part(0).x, shape.part(0).y),
                  (shape.part(3).x, shape.part(3).y),
                  ((shape.part(4).x + shape.part(3).x)/2, (shape.part(4).y+shape.part(3).y)/2),
                  (shape.part(5).x, shape.part(5).y),
                  ((shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2),
                  ((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6),
                  (shape.part(8).x, shape.part(8).y),
                  ((shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6),
                  ((shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2),
                  (shape.part(11).x, shape.part(11).y),
                  ((shape.part(13).x + shape.part(12).x)/2, (shape.part(13).y+shape.part(12).y)/2),
                  (shape.part(13).x, shape.part(13).y),
                  (shape.part(16).x, shape.part(16).y)], (int(colors[base_idx][0]),int(colors[base_idx][1]),int(colors[base_idx][2])))
    # neck 
    neck_height = shape.part(4).y - shape.part(3).y
    m = ((shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6- (shape.part(5).y+shape.part(6).y)/2) / float((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6 - (shape.part(5).x + shape.part(6).x)/2)
    b = (shape.part(5).y+shape.part(6).y)/2 - m*(shape.part(5).x + shape.part(6).x)/2
    draw.polygon([((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b),((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height),
                  ((shape.part(9).x+shape.part(10).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height), ((shape.part(9).x+shape.part(10).x)/2,shape.part(11).y)],
                   (int(colors[base_idx][0]),int(colors[base_idx][1]),int(colors[base_idx][2])))

    # nose
    nose_height = (shape.part(33).y - shape.part(30).y)/float(5) 
    draw.polygon([(shape.part(33).x, (shape.part(33).y+shape.part(30).y)/2), (shape.part(33).x+nose_height/3, (shape.part(33).y+shape.part(30).y)/2-nose_height/2), 
                    ((shape.part(33).x) - nose_height/1, (shape.part(33).y+shape.part(30).y)/2 - nose_height)],
                 (int(colors[not base_idx][0]),int(colors[not base_idx][1]),int(colors[not base_idx][2])))
  
    return im





def draw_lineart(im, shape):
    draw = ImageDraw.Draw(im)

    # Draw sides of face
    drawLine(draw, im, shape.part(0).x, shape.part(0).y, shape.part(3).x, shape.part(3).y, (0,0,0,255) )
    drawLine(draw, im, shape.part(13).x, shape.part(13).y, shape.part(16).x, shape.part(16).y, (0,0,0,255))

    # Draw cheeks
    drawLine(draw, im, shape.part(3).x, shape.part(3).y, (shape.part(4).x + shape.part(3).x)/2, (shape.part(4).y+shape.part(3).y)/2, (0,0,0,255))
    drawLine(draw, im, shape.part(5).x, shape.part(5).y, (shape.part(3).x + shape.part(4).x)/2, (shape.part(4).y+shape.part(3).y)/2, (0,0,0,255)) 

    drawLine(draw, im, shape.part(13).x, shape.part(13).y, (shape.part(13).x + shape.part(12).x)/2, (shape.part(13).y+shape.part(12).y)/2, (0,0,0,255))
    drawLine(draw, im, shape.part(11).x, shape.part(11).y, (shape.part(13).x + shape.part(12).x)/2, (shape.part(13).y+shape.part(12).y)/2, (0,0,0,255))   
    # drawLine(draw, im, shape.part(3).x, shape.part(3).y, shape.part(5).x, shape.part(5).y, (0,0,0,255)) ##
    # drawLine(draw, im, shape.part(13).x, shape.part(13).y, shape.part(11).x, shape.part(11).y, (0,0,0,255)) ##

    # Draw chin
    drawLine(draw, im, shape.part(5).x, shape.part(5).y, (shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2, (0,0,0,255))
    drawLine(draw, im, (shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6, (shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2, (0,0,0,255))  
    drawLine(draw, im, (shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6, shape.part(8).x, shape.part(8).y, (0,0,0,255))  
    
    drawLine(draw, im, shape.part(11).x, shape.part(11).y, (shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2, (0,0,0,255))
    drawLine(draw, im, (shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6, (shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2, (0,0,0,255))  
    drawLine(draw, im, (shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6, shape.part(8).x, shape.part(8).y, (0,0,0,255))  
    # drawLine(draw, im, shape.part(5).x, shape.part(5).y, shape.part(8).x, shape.part(8).y, (0,0,0,255)) ## 
    # drawLine(draw, im, shape.part(11).x, shape.part(11).y, shape.part(8).x, shape.part(8).y, (0,0,0,255)) ##

    # draw nose (for female)
    nose_height = (shape.part(33).y - shape.part(30).y)/float(5)
    drawLine(draw, im, shape.part(33).x, (shape.part(33).y+shape.part(30).y)/2, (shape.part(33).x) - nose_height/1, (shape.part(33).y+shape.part(30).y)/2 - nose_height, (0,0,0,255))

    im = draw_mouth(im, shape, draw)

    # Draw neck
    m = ((shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6- (shape.part(5).y+shape.part(6).y)/2) / float((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6 - (shape.part(5).x + shape.part(6).x)/2)
    b = (shape.part(5).y+shape.part(6).y)/2 - m*(shape.part(5).x + shape.part(6).x)/2
    
    neck_height = shape.part(4).y - shape.part(3).y
    drawLine(draw, im, (shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b, (shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height, (0,0,0,255))
    y_pos = m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height
    m = ((shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6- (shape.part(11).y+shape.part(10).y)/2) / float((shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6 - (shape.part(11).x + shape.part(10).x)/2)
    b = (shape.part(11).y+shape.part(10).y)/2 - m*(shape.part(11).x + shape.part(10).x)/2

    drawLine(draw, im, (shape.part(9).x+shape.part(10).x)/2, m*(shape.part(9).x+shape.part(10).x)/2+b, (shape.part(9).x+shape.part(10).x)/2, y_pos, (0,0,0,255))
  
    # drawLine(draw, im, (shape.part(6).x+shape.part(7).x)/2, (shape.part(6).y+shape.part(7).y)/2, (shape.part(6).x+shape.part(7).x)/2, (shape.part(6).y+shape.part(7).y)/2 + neck_height, (0,0,0,255))
    # drawLine(draw, im, (shape.part(10).x+shape.part(9).x)/2, (shape.part(10).y+shape.part(9).y)/2, (shape.part(10).x+shape.part(9).x)/2, (shape.part(10).y+shape.part(9).y)/2 + neck_height, (0,0,0,255))
    

    return im
 
def draw_mouth(im, shape, draw):
    # Need to shrink mouth horizontally (in between eyes), check if open smile/frown/etc

    # use height of nose as threshold
    threshold = shape.part(33).y - shape.part(31).y

    anime_mouth_width = shape.part(42).x - shape.part(39).x 
    real_mouth_width = shape.part(64).x - shape.part(60).x
    ratio = float(anime_mouth_width)/real_mouth_width * float(3)/4

    center_x = (shape.part(62).x + shape.part(66).x)/2
    center_y = (shape.part(62).y + shape.part(66).y)/2

    shift_x = center_x - (shape.part(62).x*ratio + shape.part(66).x*ratio)/float(2)
    shift_y = center_y - (shape.part(62).y*ratio + shape.part(66).y*ratio)/float(2)


    mouth_height = shape.part(66).y - shape.part(62).y 
    if mouth_height > threshold:
        line_range = range(61,68)
        pts = []
        for i in range(60,68):
            if i != 62:
                pts.append((shape.part(i).x*ratio + shift_x, shape.part(i).y*ratio+shift_y))
        pts[0] = (pts[0][0] + 3,pts[0][1])
        draw.polygon(pts,(255,255,255))
        drawLine(draw, im, shape.part(60).x * ratio + shift_x, shape.part(60).y * ratio + shift_y,
                 shape.part(67).x * ratio + shift_x, shape.part(67).y * ratio + shift_y, (0,0,0,255))
        

    else: 
        line_range = range(61,65)

    prev_x = shape.part(60).x * ratio + shift_x
    prev_y = shape.part(60).y * ratio + shift_y
    for i in line_range: 
        if i != 62:
            drawLine(draw, im, prev_x, prev_y, shape.part(i).x * ratio + shift_x, shape.part(i).y * ratio + shift_y, (0,0,0,255))
            prev_x = shape.part(i).x * ratio + shift_x
            prev_y = shape.part(i).y * ratio + shift_y
    
    return im



def predict_shape(fname):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL)

    img = io.imread(fname)
    dets = detector(img, 1)

    for k, d in enumerate(dets): # should only detect one face
        shape = predictor(img, d)
        return shape

def quantize_skin(fname, shape):
    img = imread(fname)
    img = np.double(img)
    img = img[shape.part(1).y: shape.part(4).y, shape.part(4).x:shape.part(12).x]

    pixel = np.reshape(img,(img.shape[0]*img.shape[1],3))         
    centroids,_ = kmeans(pixel,2)
    return centroids

#################################### IMPLEMENTATION OF XIAOLIN WU'S LINE ALGORITHM FOR ANTI-ALIASING ####################################

def plot(draw, img, x, y, c, col,steep):
    if steep:
        x,y = y,x
    if x < img.size[0] and y < img.size[1] and x >= 0 and y >= 0:
        c = c * (float(col[3])/255.0)
        p = img.getpixel((x,y))
        draw.point((int(x),int(y)),fill=(int((p[0]*(1-c)) + col[0]*c), int((p[1]*(1-c)) + col[1]*c), int((p[2]*(1-c)) + col[2]*c),255))

def iround(x):
    return ipart(x + 0.5)

def ipart(x):
    return math.floor(x)

def fpart(x):
    return x-math.floor(x)

def rfpart(x):
    return 1 - fpart(x)

def drawLine(draw, img, x1, y1, x2, y2, col):
    dx = x2 - x1
    dy = y2 - y1

    steep = abs(dx) < abs(dy)
    if steep:
        x1,y1=y1,x1
        x2,y2=y2,x2
        dx,dy=dy,dx
    if x2 < x1:
        x1,x2=x2,x1
        y1,y2=y2,y1
    gradient = float(dy) / float(dx)

    #handle first endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = xend    #this will be used in the main loop
    ypxl1 = ipart(yend)
    plot(draw, img, xpxl1, ypxl1, rfpart(yend) * xgap,col, steep)
    plot(draw, img, xpxl1, ypxl1 + 1, fpart(yend) * xgap,col, steep)
    intery = yend + gradient # first y-intersection for the main loop

    #handle second endpoint
    xend = round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = xend    # this will be used in the main loop
    ypxl2 = ipart (yend)
    plot (draw, img, xpxl2, ypxl2, rfpart (yend) * xgap,col, steep)
    plot (draw, img, xpxl2, ypxl2 + 1, fpart (yend) * xgap,col, steep)

    #main loop
    for x in range(int(xpxl1 + 1), int(xpxl2 )):
        plot (draw, img, x, ipart (intery), rfpart (intery),col, steep)
        plot (draw, img, x, ipart (intery) + 1, fpart (intery),col, steep)
        intery = intery + gradient

if __name__ == "__main__":
    if len(sys.argv) == 2:
        autoanime(sys.argv[1])
    else:
        print("Correct usage: ./autoanime file")