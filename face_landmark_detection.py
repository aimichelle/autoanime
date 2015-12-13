#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
from PIL import Image, ImageDraw

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.

        im = Image.open("me.jpg")
        draw = ImageDraw.Draw(im)
        prev_x = shape.part(0).x
        prev_y = shape.part(0).y
        for i in range(1,17):
            if (i != 1 and i != 2 and i != 4   and i != 7 and i != 9 and  i != 12  and i != 14 and i != 15):
                draw.line([prev_x, prev_y, shape.part(i).x,shape.part(i).y], width=1)
                prev_x = shape.part(i).x
                prev_y = shape.part(i).y
        prev_x = shape.part(60).x
        prev_y = shape.part(60).y
        for i in range(61,68):
            draw.line([prev_x, prev_y, shape.part(i).x,shape.part(i).y], width=1)
            prev_x = shape.part(i).x
            prev_y = shape.part(i).y
        draw.line([shape.part(60).x, shape.part(60).y, shape.part(67).x,shape.part(67).y], width=1)

        draw.line([shape.part(56).x, shape.part(56).y, shape.part(57).x,shape.part(57).y], width=1)
        draw.line([shape.part(57).x, shape.part(57).y, shape.part(58).x,shape.part(58).y], width=1)

        draw.line([shape.part(50).x, shape.part(50).y, shape.part(51).x,shape.part(51).y], width=1)
        draw.line([shape.part(51).x, shape.part(51).y, shape.part(52).x,shape.part(52).y], width=1)

        draw.line([shape.part(31).x, shape.part(31).y, shape.part(33).x,shape.part(33).y], width=1)
        draw.line([shape.part(33).x, shape.part(33).y, shape.part(35).x,shape.part(35).y], width=1)
        draw.line([shape.part(0).x, shape.part(0).y, shape.part(3).x, shape.part(3).y],width=1)
        draw.line([shape.part(3).x, shape.part(3).y, shape.part(5).x, shape.part(5).y],width=1)
        draw.line([shape.part(5).x, shape.part(5).y, shape.part(8).x, shape.part(8).y],width=1)
        draw.line([shape.part(8).x, shape.part(8).y, shape.part(11).x, shape.part(11).y],width=1)
        draw.line([shape.part(11).x, shape.part(11).y, shape.part(13).x, shape.part(13).y],width=1)
        draw.line([shape.part(13).x, shape.part(13).y, shape.part(16).x, shape.part(16).y],width=1)
        for i in range(68):
            draw.point([shape.part(i).x, shape.part(i).y])
        im.save("test.png", "PNG")
        win.add_overlay(shape)
    
    # win.add_overlay(dets)
    # dlib.hit_enter_to_continue()
