#!/usr/bin/python

import sys
import os
import math
import dlib
import glob
from skimage import io
from PIL import Image, ImageDraw

MODEL = "shape_predictor_68_face_landmarks.dat"

def autoanime(fname):
    shape = predict_shape(fname)

    orig_im = Image.open(fname)
    new_im = Image.new("RGB", (orig_im.size[0], orig_im.size[1]))

    # Draw outline
    new_im = draw_lineart(new_im, shape)

    # Save image
    new_im.save("test.png", "PNG")

def draw_lineart(im, shape):
    draw = ImageDraw.Draw(im)

    # Draw lines
    prev_x = shape.part(0).x
    prev_y = shape.part(0).y
    for i in range(1,17):
        if (i != 1 and i != 2 and i != 4   and i != 7 and i != 9 and  i != 12  and i != 14 and i != 15):
            drawLine(draw, im, prev_x, prev_y, shape.part(i).x,shape.part(i).y, (0,0,0,255))
            prev_x = shape.part(i).x
            prev_y = shape.part(i).y
    prev_x = shape.part(60).x
    prev_y = shape.part(60).y
    drawLine(draw, im, shape.part(31).x, shape.part(31).y, shape.part(33).x,shape.part(33).y, (255,255,255,255))
    drawLine(draw, im, shape.part(33).x, shape.part(33).y, shape.part(35).x,shape.part(35).y, (0,0,0,255))

    return im

def predict_shape(fname):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL)

    img = io.imread(fname)
    dets = detector(img, 1)

    for k, d in enumerate(dets): # should only detect one face
        shape = predictor(img, d)
        return shape

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