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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2gray, gray2rgb
from scipy import signal
from scipy.cluster.vq import kmeans,vq
from pylab import imread,imshow,show
import colorsys
import hair_detection

MODEL = "shape_predictor_68_face_landmarks.dat"
DEBUG_PRINT = False
GENDER = 'none'
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

#As a note, I think the optimal input size is about 700x900.
#Not sure if it works with lower quality than 500px wide due to eye resizing issues.

def autoanime(fname):

    im = Image.open(fname)
    if (im.size[1] > 700):
        print "resizing"
        a = im.resize((int(im.size[0]*float(700)/im.size[1]), 700), resample=Image.BICUBIC)
        a.save("test_resize.jpg")
        fname = "test_resize.jpg"

    shape = predict_shape(fname)
    
    orig_im = Image.open(fname)

    new_im = Image.new("RGB", (orig_im.size[0], orig_im.size[1]), color=(255,255,255))

    #hair
    mask = hair_detection.get_hair_mask(fname)
    long_hair = hair_detection.long_hair(mask,shape)
    # hair_file = hair_detection.match_hair(mask, shape, long_hair, GENDER)

    hair_color = hair_detection.hair_color(shape,fname)

    if "ncc" in hair_file:
        print "we have long hair."
        new_im = hair_detection.draw_long_hair(shape,hair_file,new_im,hair_color)

    skin_color = quantize_skin(fname,shape)
    new_im = color_skin(new_im, shape, skin_color)
    print "first half hair done!"
    
    
    # Draw outline
    new_im = draw_lineart(new_im, shape, skin_color)
    new_im = draw_forehead(new_im, shape, skin_color)
    print "lineart done! now starting eyes..."

    ## eyes ##

    new_im = process_eyes(shape, orig_im, new_im)
    print "eyes done!"


    new_im = process_eyebrows(shape, orig_im, new_im)
    print "eyebrows done!"

    #add ears
    new_im = add_ears(shape, new_im, skin_color)
    print "ears done!"


    #draw the rest of the hair
    angle = hair_detection.get_hair_angle(shape)
    new_im = hair_detection.draw_hair(shape, hair_file, new_im, angle, hair_color)

    print "hair all done!"
    print "finished! check the folder for a saved PNG."
    # Save image
    new_im.save("test.png", "PNG")


def draw_forehead(im, shape, colors):
    if (0.2126*colors[0][0] + 0.7152*colors[0][1] + 0.0722*colors[0][2]) > (0.2126*colors[1][0] + 0.7152*colors[1][1] + 0.0722*colors[1][2]):
        base_idx = 0
    else:
        base_idx = 1
    draw = ImageDraw.Draw(im)

    # face
    points = []
    for i in range(17):
        if (i <= 4 or i >= 12):
            points.append((shape.part(i).x, shape.part(0).y - (shape.part(i).y - shape.part(0).y)))
        elif i == 9:
            points.append((shape.part(i).x, shape.part(0).y - (shape.part(4).y - shape.part(0).y)))
    points.append((shape.part(16).x, shape.part(16).y ))
    draw.polygon(points, (int(colors[base_idx][0]),int(colors[base_idx][1]),int(colors[base_idx][2])))
    return im

def process_eyebrows(shape, orig_im, new_im):
    """wrapper for processing eyebrows"""
    left, right = find_eyebrows(shape, orig_im)
    print "found eyebrows!"
    are_they_bushy = render_big_eyebrows(left, right)
    print 'should we be drawing bushy eyebrows? ', are_they_bushy
    new_im = draw_eyebrows(shape, new_im, are_they_bushy)
    return new_im

def process_eyes(shape, orig_im, new_im):
    """wrapper for eye functions"""
    left, right = crop_eyes(shape, orig_im)
    left_r, right_r = resize(left, right)
    print "now matching eyes (it takes a min or two)..."
    left_a, right_a = match_anime(left_r, right_r)

    #paste eye
    left_e_w, left_e_h = left.size
    left_e_w, left_e_h = int(left_e_w*1.25), int(left_e_h*1.25) #haha expand eye

    right_e_w, right_e_h = right.size
    right_e_w, right_e_h = int(right_e_w*1.25), int(right_e_h*1.25) 

    lratio = 200./left_e_w
    l_anime_eye_resized = left_a.resize((left_e_w, int(200./lratio)) , resample=Image.BICUBIC)

    rratio = 200./right_e_w
    r_anime_eye_resized = right_a.resize((right_e_w, int(200./rratio)) , resample=Image.BICUBIC)

    if DEBUG_PRINT:
        print 'original left eye: ', left.size
        print 'original right eye: ', right.size
        print 'l/r ratio: ', lratio, rratio
        print 'new left anime size: ', left_e_w, left_e_h*lratio
        print 'new right anime size: ', right_e_w, right_e_h*rratio
        print 'left proportion (should be 1)', left_e_w/(left_e_h*lratio)

    ly = (shape.part(37).y + shape.part(38).y)/2
    lx = (shape.part(37).x + shape.part(38).x)/2

    ry = (shape.part(43).y + shape.part(44).y)/2
    rx = (shape.part(43).x + shape.part(44).x)/2

    new_l_w, new_l_h = l_anime_eye_resized.size
    new_r_w, new_r_h = r_anime_eye_resized.size

    if DEBUG_PRINT:
        l_anime_eye_resized.show()
        r_anime_eye_resized.show()

    new_im.paste(l_anime_eye_resized, box=((lx - new_l_w/2 - new_l_w/12, int(ly-new_l_h/9))), mask=l_anime_eye_resized)
    new_im.paste(r_anime_eye_resized, box=((rx - new_r_w/2 + new_r_w/12, int(ry-new_r_h/9))), mask=r_anime_eye_resized)

    return new_im




def color_skin(im, shape, colors):
    # Find darkest and lightest color using luminance
    if (0.2126*colors[0][0] + 0.7152*colors[0][1] + 0.0722*colors[0][2]) > (0.2126*colors[1][0] + 0.7152*colors[1][1] + 0.0722*colors[1][2]):
        base_idx = 0
    else:
        base_idx = 1 #baseidx is the lighter color.
    draw = ImageDraw.Draw(im)

     # neck 
    neck_height = shape.part(4).y - shape.part(3).y +10
    m = ((shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6- (shape.part(5).y+shape.part(6).y)/2) / float((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6 - (shape.part(5).x + shape.part(6).x)/2)
    b = (shape.part(5).y+shape.part(6).y)/2 - m*(shape.part(5).x + shape.part(6).x)/2
    draw.polygon([((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b),((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height),
                  ((shape.part(9).x+shape.part(10).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height), ((shape.part(9).x+shape.part(10).x)/2,shape.part(11).y)],
                  (int(colors[base_idx][0]),int(colors[base_idx][1]),int(colors[base_idx][2])))

    # neck shadow
    draw.polygon([((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b),((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height/1.7),
                  ((shape.part(9).x+shape.part(10).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height/1.7), ((shape.part(9).x+shape.part(10).x)/2,shape.part(11).y)],  
                (int(colors[not base_idx][0]),int(colors[not base_idx][1]),int(colors[not base_idx][2])))
    # draw.polygon([((shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2+neck_height/4.),
    #               ((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6+neck_height/4.),
    #               (shape.part(8).x, shape.part(8).y+neck_height/4.),
    #               ((shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6+neck_height/4.),
    #               ((shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2+neck_height/4.)],
    #               (int(colors[not base_idx][0]),int(colors[not base_idx][1]),int(colors[not base_idx][2])))

    # draw.polygon([((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b-neck_height),
    #                 ((shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height),
    #                 ((shape.part(6).x+shape.part(7).x)/2-100, m*(shape.part(6).x+shape.part(7).x)/2+b+neck_height),
    #                 ((shape.part(6).x+shape.part(7).x)/2-100, m*(shape.part(6).x+shape.part(7).x)/2+b-neck_height)], (255,255,255))

    # draw.polygon([((shape.part(9).x+shape.part(10).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b-neck_height),
    #             ((shape.part(9).x+shape.part(10).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height),
    #             ((shape.part(9).x+shape.part(10).x)/2+100, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height),
    #             ((shape.part(9).x+shape.part(10).x)/2+100, m*(shape.part(6).x+shape.part(7).x)/2 + b-neck_height)], (255,255,255))

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
   

    # nose
    pts = []
    y_const = 15

    nose_height = (shape.part(33).y - shape.part(30).y)/float(5) 
    pts.append( (shape.part(33).x - nose_height/1, (shape.part(33).y+shape.part(30).y)/2 - nose_height + y_const))
    pts.append((shape.part(33).x, (shape.part(33).y + shape.part(30).y)/2.+nose_height/2 + y_const))
    # pts.append((shape.part(30).x, shape.part(30).y))
    pts.append(((shape.part(34).x+shape.part(33).x)/2-nose_height*1.5, shape.part(30).y + y_const))
    
    # for i in range(33,36):
    #     pts.append((shape.part(i).x, shape.part(i).y))

    draw.polygon(pts,(int(colors[not base_idx][0]),int(colors[not base_idx][1]),int(colors[not base_idx][2])))
  
    return im

def draw_eyebrows(shape, im, bushy):
    '''draws eyebrows'''    
    draw = ImageDraw.Draw(im)

    l_eyebrow = [shape.part(17),shape.part(18),shape.part(19),shape.part(20),shape.part(21)]
    r_eyebrow = [shape.part(22),shape.part(23),shape.part(24),shape.part(25),shape.part(26)]
    if not bushy: #draw lines
        for i in xrange(len(l_eyebrow)-1):
           drawLineWithStroke(1, draw, im, l_eyebrow[i].x, l_eyebrow[i].y, l_eyebrow[i+1].x, l_eyebrow[i+1].y, (0,0,0,255) )
           drawLineWithStroke(1, draw, im, r_eyebrow[i].x, r_eyebrow[i].y, r_eyebrow[i+1].x, r_eyebrow[i+1].y, (0,0,0,255) )
    else: #draw polygon
        xy_left = []
        xy_right = []
        for i in xrange(len(l_eyebrow)):
           xy_left.append((l_eyebrow[i].x, l_eyebrow[i].y))
           xy_right.append((r_eyebrow[i].x, r_eyebrow[i].y))

        draw.polygon(xy_left, (0,0,0), (0,0,0)) #TODO: fill (first one) is hair color
        draw.polygon(xy_right, (0,0,0), (0,0,0))

    return im


def draw_lineart(im, shape, colors):
    draw = ImageDraw.Draw(im)
    s_width = 2

    # Draw sides of face
    drawLineWithStroke(s_width, draw, im, shape.part(0).x, shape.part(0).y, shape.part(3).x, shape.part(3).y, (0,0,0,255) )
    drawLineWithStroke(s_width, draw, im, shape.part(13).x, shape.part(13).y, shape.part(16).x, shape.part(16).y, (0,0,0,255))

    # Draw cheeks
    drawLineWithStroke(s_width, draw, im, shape.part(3).x, shape.part(3).y, (shape.part(4).x + shape.part(3).x)/2, (shape.part(4).y+shape.part(3).y)/2, (0,0,0,255))
    drawLineWithStroke(s_width, draw, im, shape.part(5).x, shape.part(5).y, (shape.part(3).x + shape.part(4).x)/2, (shape.part(4).y+shape.part(3).y)/2, (0,0,0,255)) 

    drawLineWithStroke(s_width, draw, im, shape.part(13).x, shape.part(13).y, (shape.part(13).x + shape.part(12).x)/2, (shape.part(13).y+shape.part(12).y)/2, (0,0,0,255))
    drawLineWithStroke(s_width, draw, im, shape.part(11).x, shape.part(11).y, (shape.part(13).x + shape.part(12).x)/2, (shape.part(13).y+shape.part(12).y)/2, (0,0,0,255))   
    # drawLine(draw, im, shape.part(3).x, shape.part(3).y, shape.part(5).x, shape.part(5).y, (0,0,0,255)) ##
    # drawLine(draw, im, shape.part(13).x, shape.part(13).y, shape.part(11).x, shape.part(11).y, (0,0,0,255)) ##

    # Draw chin
    drawLineWithStroke(s_width, draw, im, shape.part(5).x, shape.part(5).y, (shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2, (0,0,0,255))
    drawLineWithStroke(s_width, draw, im, (shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6, (shape.part(5).x + shape.part(6).x)/2, (shape.part(5).y+shape.part(6).y)/2, (0,0,0,255))  
    drawLineWithStroke(s_width, draw, im, (shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6, shape.part(8).x, shape.part(8).y, (0,0,0,255))  
    
    drawLineWithStroke(s_width, draw, im, shape.part(11).x, shape.part(11).y, (shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2, (0,0,0,255))
    drawLineWithStroke(s_width, draw, im, (shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6, (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6, (shape.part(11).x + shape.part(10).x)/2, (shape.part(11).y+shape.part(10).y)/2, (0,0,0,255))  
    drawLineWithStroke(s_width, draw, im, (shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6,  (shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6, shape.part(8).x, shape.part(8).y, (0,0,0,255))  
    # drawLine(draw, im, shape.part(5).x, shape.part(5).y, shape.part(8).x, shape.part(8).y, (0,0,0,255)) ## 
    # drawLine(draw, im, shape.part(11).x, shape.part(11).y, shape.part(8).x, shape.part(8).y, (0,0,0,255)) ##

    # draw nose (for female)
    move_down = 20
    nose_height = (shape.part(33).y - shape.part(30).y)/float(5)
    drawLineWithStroke(s_width, draw, im, shape.part(33).x, (shape.part(33).y+shape.part(30).y)/2 + move_down, (shape.part(33).x) - nose_height/1, (shape.part(33).y+shape.part(30).y)/2 - nose_height + move_down, (0,0,0,255))

    im = draw_mouth(im, shape, draw, colors)

    # Draw neck
    m = ((shape.part(8).y)+(shape.part(7).y-shape.part(8).y)/6- (shape.part(5).y+shape.part(6).y)/2) / float((shape.part(8).x)+(shape.part(7).x-shape.part(8).x)/6 - (shape.part(5).x + shape.part(6).x)/2)
    b = (shape.part(5).y+shape.part(6).y)/2 - m*(shape.part(5).x + shape.part(6).x)/2
    
    neck_height = shape.part(4).y - shape.part(3).y + 10
    drawLineWithStroke(s_width, draw, im, (shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2+b, (shape.part(6).x+shape.part(7).x)/2, m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height, (0,0,0,255))
    y_pos = m*(shape.part(6).x+shape.part(7).x)/2 + b+ neck_height
    m = ((shape.part(8).y)+(shape.part(9).y-shape.part(8).y)/6- (shape.part(11).y+shape.part(10).y)/2) / float((shape.part(8).x)+(shape.part(9).x-shape.part(8).x)/6 - (shape.part(11).x + shape.part(10).x)/2)
    b = (shape.part(11).y+shape.part(10).y)/2 - m*(shape.part(11).x + shape.part(10).x)/2

    drawLineWithStroke(s_width, draw, im, (shape.part(9).x+shape.part(10).x)/2, m*(shape.part(9).x+shape.part(10).x)/2+b, (shape.part(9).x+shape.part(10).x)/2, y_pos, (0,0,0,255))
  
    # drawLine(draw, im, (shape.part(6).x+shape.part(7).x)/2, (shape.part(6).y+shape.part(7).y)/2, (shape.part(6).x+shape.part(7).x)/2, (shape.part(6).y+shape.part(7).y)/2 + neck_height, (0,0,0,255))
    # drawLine(draw, im, (shape.part(10).x+shape.part(9).x)/2, (shape.part(10).y+shape.part(9).y)/2, (shape.part(10).x+shape.part(9).x)/2, (shape.part(10).y+shape.part(9).y)/2 + neck_height, (0,0,0,255))
    

    return im
 
def draw_mouth(im, shape, draw, colors):
    if (0.2126*colors[0][0] + 0.7152*colors[0][1] + 0.0722*colors[0][2]) > (0.2126*colors[1][0] + 0.7152*colors[1][1] + 0.0722*colors[1][2]):
        base_idx = 0
    else:
        base_idx = 1
    shadow = (int(colors[not base_idx][0]),int(colors[not base_idx][1]),int(colors[not base_idx][2]))
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
        # teeth shadow
        prev_x = shape.part(60).x * ratio + shift_x 
        prev_y = shape.part(60).y * ratio + shift_y + 2.5
        for j in range(61,65):
            drawLineWithStroke(1,draw,im,prev_x, prev_y,shape.part(j).x*ratio+shift_x, shape.part(j).y*ratio+shift_y+2.5, (150,150,150,255))
            prev_x = shape.part(j).x * ratio + shift_x 
            prev_y = shape.part(j).y * ratio + shift_y + 2.5
        drawLineWithStroke(1, draw, im, shape.part(60).x * ratio + shift_x, shape.part(60).y * ratio + shift_y,
                 shape.part(67).x * ratio + shift_x, shape.part(67).y * ratio + shift_y, (0,0,0,255))
        

    else: 
        line_range = range(61,65)
        mouth_height = shape.part(57).y - shape.part(51).y 

    prev_x = shape.part(60).x * ratio + shift_x
    prev_y = shape.part(60).y * ratio + shift_y
    for i in line_range: 
        if i != 62:
            drawLineWithStroke(1, draw, im, prev_x, prev_y, shape.part(i).x * ratio + shift_x, shape.part(i).y * ratio + shift_y, (0,0,0,255))
            prev_x = shape.part(i).x * ratio + shift_x
            prev_y = shape.part(i).y * ratio + shift_y

    pts = []
    prev_x = shape.part(56).x * ratio + shift_x
    prev_y = shape.part(56).y * ratio + shift_y
    pts.append((prev_x,prev_y))
    for i in range(57,59):
        drawLineWithStroke(1, draw, im, prev_x, prev_y, shape.part(i).x * ratio + shift_x, shape.part(i).y * ratio + shift_y, (shadow[0], shadow[1], shadow[2],255))
        prev_x = shape.part(i).x * ratio + shift_x
        prev_y = shape.part(i).y * ratio + shift_y

        pts.append((prev_x,prev_y))
    pts.append((shape.part(57).x*ratio+shift_x, shape.part(57).y*ratio+shift_y+mouth_height/8.))
    draw.polygon(pts, shadow)
    return im

def add_ears(shape, im, colors):
    if (0.2126*colors[0][0] + 0.7152*colors[0][1] + 0.0722*colors[0][2]) > (0.2126*colors[1][0] + 0.7152*colors[1][1] + 0.0722*colors[1][2]):
        base_idx = 0
    else:
        base_idx = 1 #baseidx is the lighter color.
    ear_color = colors[base_idx]
    #print 'rgb color is ', ear_color
    ear_color = [c/255. for c in ear_color]
    ear_color_hsv = colorsys.rgb_to_hsv(ear_color[0], ear_color[1], ear_color[2])
    print 'hsv color is ', ear_color_hsv
    hue = ear_color_hsv[0] * 360
    #change color
    left_ear = Image.open('ears/left.png')
    right_ear = Image.open('ears/right.png')
    left_ear = colorize(left_ear, hue)
    right_ear = colorize(right_ear, hue)

    left_angle, right_angle = get_ear_angles(shape)
    print 'rotating left ear', left_angle, ' and right ear', right_angle
    left_ear = left_ear.rotate(left_angle, resample=Image.BICUBIC, expand = 1)
    right_ear = right_ear.rotate(right_angle, resample=Image.BICUBIC, expand = 1)

    left_height = abs(shape.part(17).y - shape.part(2).y)
    right_height = abs(shape.part(26).y - shape.part(14).y)
    lw, lh = left_ear.size
    rw, rh = right_ear.size 
    left_ear = left_ear.resize((int(lw*left_height/lh), left_height), resample=Image.BICUBIC)
    right_ear = right_ear.resize((int(rw*right_height/rh), right_height), resample=Image.BICUBIC)
    new_lw, new_lh = left_ear.size
    new_rw, new_rh = right_ear.size
    print new_lw, new_lh

    #reverse paste for layers!
    #make mask out of original image
    mask = np.asarray(im)
    mask.setflags(write=True)
    mask[mask == 255] = 250
    mask[mask != 250] = 255
    mask[mask != 255] = 0
    kernel = np.ones((1,1),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 2)
    mask_im = Image.fromarray(np.uint8(dilation)).convert('L')

    new_im = Image.new("RGB", (mask_im.size[0], mask_im.size[1]), color=(255,255,255))

    new_im.paste(left_ear, box=((shape.part(0).x-new_lw+new_lw/8, shape.part(17).y+new_lh/5)), mask=left_ear)
    new_im.paste(right_ear, box=((shape.part(16).x-new_rw/3, shape.part(26).y+new_lh/6)), mask=right_ear)

    new_im.paste(im, box=((0,0)), mask=mask_im)

    return new_im


def predict_shape(fname):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL)

    img = io.imread(fname)
    dets = detector(img, 1)

    for k, d in enumerate(dets): # should only detect one face
        shape = predictor(img, d)
        return shape

def dist(p1,p2):
    return math.sqrt(sum([pow(c1-c2,2) for (c1,c2) in zip(p1,p2)]))

def crop_wrapper(x, y, x_plus, y_plus, im):
    """crops image centered at (x,y) by (x_plus, y_plus) in each direction."""
    return im.crop((x-x_plus, y-y_plus, x+x_plus, y+y_plus))

def find_eyebrows(shape, orig_im):
    """returns cropped eyebrows."""
    
    l_eyebrow = [shape.part(17),shape.part(18),shape.part(19),shape.part(20),shape.part(21)]
    r_eyebrow = [shape.part(22),shape.part(23),shape.part(24),shape.part(25),shape.part(26)]

    min_left_eyebrow = min([p.y for p in l_eyebrow])
    min_right_eyebrow = min([p.y for p in r_eyebrow])

    max_left_eye = max(shape.part(37).y, shape.part(38).y)
    max_right_eye = max(shape.part(43).y, shape.part(44).y)

    left_offset = abs(min_left_eyebrow - max_left_eye)
    right_offset = abs(min_right_eyebrow - max_right_eye)

    left_x_offset = max(abs(shape.part(19).x - shape.part(17).x), abs(shape.part(19).x - shape.part(21).x))
    right_x_offset = max(abs(shape.part(24).x - shape.part(22).x), abs(shape.part(24).x - shape.part(26).x))

    left_offset = int(left_offset / 1.5)
    right_offset = int(right_offset / 1.5)

    cropped_left = crop_wrapper(shape.part(19).x, shape.part(19).y, left_x_offset, left_offset, orig_im)
    cropped_right = crop_wrapper(shape.part(24).x, shape.part(24).y, right_x_offset, right_offset, orig_im)

    return cropped_left, cropped_right

def render_big_eyebrows(left, right): 
    """returns true/false if big eyebrows should be drawn or not."""
    left = np.asarray(left)
    right = np.asarray(right)
    l_gradient = np.gradient(rgb2gray(left))
    r_gradient = np.gradient(rgb2gray(right))

    energy_1 = np.zeros((left.shape[0], left.shape[1]))
    energy_r = np.zeros((right.shape[0], right.shape[1]))
    energy_l = abs(l_gradient[0]) 
    energy_r = abs(r_gradient[0]) 
    
    # if DEBUG_PRINT:
    #     plt.imshow(energy_l,  cmap=cm.Greys_r)
    #     plt.show()
    #     plt.imshow(left)
    #     plt.show()
    #     plt.imshow(energy_r,  cmap=cm.Greys_r)
    #     plt.show()
    #     plt.imshow(right)
    #     plt.show()
    
    h, w = energy_l.shape
    mid_h = h / 2
    #not sure about these thresholds
    cropped_l = energy_l[mid_h-(h/4):mid_h+(h/4), :]
    cropped_r = energy_r[mid_h-(h/4):mid_h+(h/4), :]

    
    s_l = cropped_l.sum() / (cropped_l.shape[0] * cropped_l.shape[1]) 
    s_r = cropped_r.sum() / (cropped_r.shape[0] * cropped_r.shape[1])
    fin = (s_l + s_r) / 2. 
    print 'eyebrow energy was ', fin
    if fin < 0.022:
        return False
    else:
        return True
    

def crop_eyes(shape, orig_im):
    """crops eyes, and rescales them to be 110x220."""    
    l_eye = [shape.part(36),shape.part(37),shape.part(38),shape.part(39),shape.part(40),shape.part(41)]
    r_eye = [shape.part(42),shape.part(43),shape.part(44),shape.part(45),shape.part(46),shape.part(47)]

    l_x_min = min([p.x for p in l_eye])
    l_x_max = max([p.x for p in l_eye])
    l_y_min = min([p.y for p in l_eye])
    l_y_max = max([p.y for p in l_eye])

    r_x_min = min([p.x for p in r_eye])
    r_x_max = max([p.x for p in r_eye])
    r_y_min = min([p.y for p in r_eye])
    r_y_max = max([p.y for p in r_eye])

    cropped_l_eye = orig_im.crop((l_x_min-10, l_y_min-10, l_x_max+10, l_y_max+10))
    cropped_r_eye = orig_im.crop((r_x_min-10, r_y_min-10, r_x_max+10, r_y_max+10))
    return cropped_l_eye, cropped_r_eye

def resize(left, right):
    """resizes both eyes to be 110x220, returns them as np.arrays"""
    wl, hl = left.size
    lratio = 220./wl
    wr, hr = right.size
    rratio = 220./wr

    left = left.resize((220, int(hl*lratio)) , resample=Image.BICUBIC)
    right = right.resize((220, int(hr*rratio)) , resample=Image.BICUBIC)
    
    _, new_h_left = left.size
    _, new_h_right = right.size

    left = np.asarray(left)
    right = np.asarray(right)
    
    #er always assume > 110 px now...TODO: increase crop if problem
    crop_l = (new_h_left - 110)/2
    crop_r = (new_h_right - 110)/2    
    if (crop_l > 0):
        left = left[crop_l:new_h_left-crop_l , :]
    if (crop_r > 0):
        right = right[crop_r:new_h_right-crop_r , :]
    return left, right        

def match_anime(left_r, right_r):
    """returns 2 anime eyes! input: resized real eyes, runs NCC (kinda slow)"""
    left_scores = []
    right_scores = []
    #normalize
    left_r_bw = rgb2gray(left_r)
    left_r_normed = (left_r_bw - left_r_bw.mean()) / left_r_bw.std()

    right_r_bw = rgb2gray(right_r)
    right_r_normed = (right_r_bw - right_r_bw.mean()) / right_r_bw.std()
    left_files = []
    right_files = []

    for filename in os.listdir('eyes/real'):
        if filename == '.DS_Store':
            continue        
        if "l" in filename:
            left_files.append(filename)
            l_temp = plt.imread('eyes/real/'+filename)
            l_temp_bw = rgb2gray(l_temp)
            l_norm = (l_temp_bw - l_temp_bw.mean()) / l_temp_bw.std()
            ncc = signal.correlate2d(left_r_normed, l_norm, mode='same')
            if GENDER == 'f':
                if 'f' in filename:
                    ncc *= 1.1
            if GENDER == 'm':
                if 'm' in filename:
                    ncc *= 1.1
            left_scores.append(ncc)
            if DEBUG_PRINT:
                print filename, ' has a score of ', ncc.max()
        if "r" in filename:
            right_files.append(filename)
            r_temp = plt.imread('eyes/real/'+filename)
            r_temp_bw = rgb2gray(r_temp)
            r_norm = (r_temp_bw - r_temp_bw.mean()) / r_temp_bw.std()
            ncc = signal.correlate2d(right_r_normed, r_norm, mode='same')
            if GENDER == 'f':
                if 'f' in filename:
                    ncc *= 1.1
            if GENDER == 'm':
                if 'm' in filename:
                    ncc *= 1.1
            right_scores.append(ncc)
            if DEBUG_PRINT:
                print filename, ' has a score of ', ncc.max()     

    #hopefully they both match...        
    i_l = np.argmax([ncc.max() for ncc in left_scores])
    i_r = np.argmax([ncc.max() for ncc in right_scores])

    #returns anime eyes...
    if (i_l != i_r):
        print "uh oh, anime eyes don't match..."
        print "left: ", left_files[i_l], " right: ", right_files[i_r]
        left_score = np.amax([ncc.max() for ncc in left_scores])
        right_score = np.amax([ncc.max() for ncc in left_scores])
        print "left score, right score: ", left_score, right_score
        if left_score > right_score:
            print "using left eye for both."
            left_anime = Image.open('eyes/anime/'+left_files[i_l])
            right_anime = Image.open('eyes/anime/'+right_files[i_l])
        elif left_score == right_score and GENDER != 'none':
            print "they're equal, so we're using the correct gender."
            use = 'left'
            if GENDER == 'm':
                if 'm' in left_files[i_l]:
                    use = 'left'
                elif 'm' in right_files[i_r]:
                    use = 'right'
            elif GENDER == 'f':
                if 'f' in left_files[i_l]:
                    use = 'left'
                elif 'f' in right_files[i_r]:
                    use = 'right'
            if use == 'left':
                print "we decided to use left."
                left_anime = Image.open('eyes/anime/'+left_files[i_l])
                right_anime = Image.open('eyes/anime/'+right_files[i_l])
            else:
                print "we decided to use right."
                left_anime = Image.open('eyes/anime/'+left_files[i_r])
                right_anime = Image.open('eyes/anime/'+right_files[i_r])
        else:
            print "using right eye for both."
            left_anime = Image.open('eyes/anime/'+left_files[i_r])
            right_anime = Image.open('eyes/anime/'+right_files[i_r])
    else:
        left_anime = Image.open('eyes/anime/'+left_files[i_l])
        right_anime = Image.open('eyes/anime/'+right_files[i_r])

    if DEBUG_PRINT:
        print left_files[i_l]
        print right_files[i_r]

        print 'the i\'s are ', i_l, i_r
        image_l = plt.imread('eyes/real/'+left_files[i_l])
        image_r = plt.imread('eyes/real/'+right_files[i_r])
        plt.imshow(left_r)
        plt.show()
        plt.imshow(image_l)
        plt.show()

        plt.imshow(right_r)
        plt.show()
        plt.imshow(image_r)
        plt.show()

    return left_anime, right_anime 


def get_ear_angles(shape):
    """returns how much we need to rotate the ears based on face orientation."""
    lrise = shape.part(3).y - shape.part(0).y
    lrun = shape.part(3).x - shape.part(0).x
    print 'left ear slope is ', float(lrise/lrun)
    left_angle = math.degrees(math.atan(float(lrise/lrun)))
    rrise = shape.part(16).y - shape.part(13).y
    rrun = shape.part(16).x - shape.part(13).x
    print 'right ear slope is ', float(rrise/rrun)
    right_angle = math.degrees(math.atan(float(rrise/rrun)))

    return -(left_angle-90), -(right_angle+90)



def quantize_skin(fname, shape):
    img = imread(fname)
    img = np.double(img)
    img = img[shape.part(1).y: shape.part(4).y, shape.part(4).x:shape.part(12).x]

    pixel = np.reshape(img,(img.shape[0]*img.shape[1],3))         
    centroids,_ = kmeans(pixel,2)
    new_centroids = []
    for centroid in centroids:
        h,l,s = colorsys.rgb_to_hls(centroid[0]/255., centroid[1]/255., centroid[2]/255.)
        l = l + .1
        r,g,b = colorsys.hls_to_rgb(h,l,s)

        new_centroids.append([r*255, g*255, b*255])
    return new_centroids


# wrapper to draw thicker lines #
def drawLineWithStroke(thickness, draw, img, x1, y1, x2, y2, col):
    """wrapper to drawLine which calls it multiple times for thicker stroke widths.
        thickness = 0 -> normal thin line 
        thickness = 1 -> medium
        thickness = 2 -> big
    """
    if thickness == 0:
        drawLine(draw, img, x1, y1, x2, y2, col)
    elif thickness == 1:
        drawLine(draw, img, x1, y1, x2, y2, col)
        drawLine(draw, img, x1-1, y1, x2-1, y2, col)
        drawLine(draw, img, x1+1, y1, x2+1, y2, col)
        drawLine(draw, img, x1, y1-1, x2, y2-1, col)
        drawLine(draw, img, x1, y1+1, x2, y2+1, col)
    elif thickness == 2:
        drawLine(draw, img, x1, y1, x2, y2, col)
        drawLine(draw, img, x1-1, y1, x2-1, y2, col)
        drawLine(draw, img, x1+1, y1, x2+1, y2, col)
        drawLine(draw, img, x1, y1-1, x2, y2-1, col)
        drawLine(draw, img, x1, y1+1, x2, y2+1, col)

        drawLine(draw, img, x1-1, y1-1, x2-1, y2-1, col)
        drawLine(draw, img, x1+1, y1-1, x2+1, y2-1, col)
        drawLine(draw, img, x1+1, y1-1, x2+1, y2-1, col)
        drawLine(draw, img, x1-1, y1+1, x2-1, y2+1, col)

        drawLine(draw, img, x1-2, y1, x2-2, y2, col)
        drawLine(draw, img, x1+2, y1, x2+2, y2, col)
        drawLine(draw, img, x1, y1-2, x2, y2-2, col)
        drawLine(draw, img, x1, y1+2, x2, y2+2, col)       
    else:
        print 'invalid thickness specified for drawLine!'



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


# hue shifting code courtesy of http://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img


if __name__ == "__main__":
    if len(sys.argv) == 2:
        autoanime(sys.argv[1])
    elif len(sys.argv) == 4:
        GENDER = sys.argv[3]
        print "starting with gender ", GENDER 
        autoanime(sys.argv[1])
    else:
        print("Correct usage: ./autoanime file")