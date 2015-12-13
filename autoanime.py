#!/usr/bin/python

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

MODEL = "shape_predictor_68_face_landmarks.dat"
DEBUG_PRINT = False

#As a note, I think the optimal input size is about 500x700.
#Not sure if it works with lower quality than that. Lol.

def autoanime(fname):
    shape = predict_shape(fname)

    orig_im = Image.open(fname)
    new_im = Image.new("RGB", (orig_im.size[0], orig_im.size[1]), color=(255,255,255))

    # test = orig_im.resize((orig_im.size[0]/100, orig_im.size[1]/100))
    # test = orig_im.resize((orig_im.size[0]*100, orig_im.size[1]*100))
    # test.save("test_resize.png", "PNG")
    # Draw outline
    new_im = draw_lineart(new_im, shape)
    # new_im = draw_lineart(orig_im, shape)
    print "lineart done! now starting eyes..."

    ## eyes ##
    new_im = process_eyes(shape, orig_im, new_im)
    print "eyes done!"

    ## eyebrows ##
    process_eyebrows(shape, orig_im)
    print "eyebrows done!"

    # Save image
    new_im.save("test-2.png", "PNG")

def process_eyebrows(shape, orig_im):
    """wrapper for processing eyebrows"""
    left, right = find_eyebrows(shape, orig_im)
    print "found eyebrows!"
    are_they_bushy = render_big_eyebrows(left, right)
    print 'should we be drawing bushy eyebrows? ', are_they_bushy
    #if !are_they_bushy: #draw normally
        #blah code here, but for now it's after the comments
    #else: 
        #figure out big line width
    #how does drawline work, michelle?

def process_eyes(shape, orig_im, new_im):
    """wrapper for eye functions"""
    left, right = crop_eyes(shape, orig_im)
    left_r, right_r = resize(left, right)
    print "now matching eyes (it takes a min or two)..."
    left_a, right_a = match_anime(left_r, right_r)

    #paste eye
    left_e_w, left_e_h = left.size
    left_e_w = int(left_e_w*1.25) #haha expand eye

    right_e_w, right_e_h = right.size
    right_e_w = int(right_e_w*1.25) 

    lratio = 200./left_e_w
    l_anime_eye_resized = left_a.resize((left_e_w, int(left_e_h*lratio)) , resample=Image.BICUBIC)

    rratio = 200./right_e_w
    r_anime_eye_resized = right_a.resize((right_e_w, int(right_e_h*rratio)) , resample=Image.BICUBIC)

    ly = (shape.part(37).y + shape.part(38).y)/2
    lx = (shape.part(37).x + shape.part(38).x)/2

    ry = (shape.part(43).y + shape.part(44).y)/2
    rx = (shape.part(43).x + shape.part(44).x)/2

    new_l_w, new_l_h = l_anime_eye_resized.size
    new_r_w, new_r_h = r_anime_eye_resized.size


    new_im.paste(l_anime_eye_resized, box=((lx - new_l_w/2 - new_l_w/12, int(ly-new_l_h/3.5))), mask=l_anime_eye_resized)
    new_im.paste(r_anime_eye_resized, box=((rx - new_r_w/2 + new_r_w/12, int(ry-new_r_h/3.5))), mask=r_anime_eye_resized)

    return new_im

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
    
    if DEBUG_PRINT:
        plt.imshow(energy_l,  cmap=cm.Greys_r)
        plt.show()
        plt.imshow(left)
        plt.show()
        plt.imshow(energy_r,  cmap=cm.Greys_r)
        plt.show()
        plt.imshow(right)
        plt.show()
    
    h, w = energy_l.shape
    mid_h = h / 2
    #not sure about these thresholds
    cropped_l = energy_l[mid_h-(h/4):mid_h+(h/4), :]
    cropped_r = energy_r[mid_h-(h/4):mid_h+(h/4), :]

    
    s_l = cropped_l.sum() / (cropped_l.shape[0] * cropped_l.shape[1]) 
    s_r = cropped_r.sum() / (cropped_r.shape[0] * cropped_r.shape[1])
    fin = (s_l + s_r) / 2. 
    print 'eyebrow energy was ', fin
    if fin < 0.02:
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
            left_scores.append(ncc)
            #print filename, ' has a score of ', ncc.max()
        if "r" in filename:
            right_files.append(filename)
            r_temp = plt.imread('eyes/real/'+filename)
            r_temp_bw = rgb2gray(r_temp)
            r_norm = (r_temp_bw - r_temp_bw.mean()) / r_temp_bw.std()
            ncc = signal.correlate2d(right_r_normed, r_norm, mode='same')
            right_scores.append(ncc)
            #print filename, ' has a score of ', ncc.max()     

    #hopefully they both match...        
    i_l = np.argmax([ncc.max() for ncc in left_scores])
    i_r = np.argmax([ncc.max() for ncc in right_scores])

    #returns anime eyes...
    if (i_l != i_r):
        print "uh oh, anime eyes don't match..."
        print "left: ", left_files[i_l], " right: ", right_files[i_r]
        left_score = np.amax([ncc.max() for ncc in left_scores])
        right_score = np.amax([ncc.max() for ncc in left_scores])
        if left_score > right_score:
            print "using left eye for both."
            left_anime = Image.open('eyes/anime/'+left_files[i_l])
            right_anime = Image.open('eyes/anime/'+right_files[i_l])
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

        print 'the is are ', i_l, i_r
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