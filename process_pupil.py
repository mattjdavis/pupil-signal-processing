# -*- coding: utf-8 -*-
"""
TO DO:
-get user input for start center (hardcoded now)
-could find artifact automatically by taking the disribution of values, and picking all the brightest
(make them all black, or set them to the mean of darkest values.)
"""

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
#NOT SURE ABOUT BACKEND

import itertools
import sys
import tifffile as tif
from tifffile import TiffWriter
import cv2

import numpy as np
import copy
import os.path
from scipy.signal import savgol_filter
import pandas as pd
from pandas.stats.moments import \
    rolling_window, rolling_apply, rolling_quantile
import time
import math
import cPickle as pickle
import argparse



'''
Defaults from mjd511

LOW_PIXEL_THRESH=130
HIGH_PIXEL_VALUE= 255
UPPER_BOUND = 50
LOWER_BOUND = 145
LEFT_BOUND = 50
RIGHT_BOUND = 160
MIN_AREA = 900
MAX_AREA =5500
'''
# python processPupil.py -f C:\\Users\\Matt\\Desktop\\CREB9\\mjd536_20180614160120.tif -p
class params:
    LOW_PIXEL_THRESH=70# its inverted
    HIGH_PIXEL_VALUE= 255
    ARTIFACT_LOW_VALUE=190 
    ARTIFACT_HIGH_VALUE=255
    UPPER_BOUND = 27#32
    LOWER_BOUND = 163#145
    LEFT_BOUND = 7#32
    RIGHT_BOUND = 158 #155
    MIN_AREA = 900
    MAX_AREA =5500
    MAX_CENTER_SHIFT= 12
    BLUR_SIZE = (3, 3)
    PUPIL_FRAME_RATE = 30.0

def calculate_baseline(data,plot=False):
    imgPeriod=1.0/params.PUPIL_FRAME_RATE
    t1=int( .1 / imgPeriod) #works well for pupil mean, 30hz
    t2=int( 10 / imgPeriod) #works well for pupil mean, 30hz
    
    data2=pd.Series(data)
    data2=rolling_window(data2.T, window=t1, win_type='boxcar',
                    min_periods=t1 / 3, center=True, axis=0).T
    data2=rolling_apply(data2.T, t2, np.nanmin, min_periods=t2 / 3,
                    center=True).T
    baseline=data2.values

    if plot:
        fig, ax = plt.subplots(2, 3,figsize=(8,6))
        ax.plot(data2)
        ax[0].set_title('Mean intesity per frame')
        ax.plot(baseline)
        ax[1].set_title('Baseline')
        plt.show()

    return baseline

def detect_blinks(images):
    mean_int=np.mean(np.mean(images,axis=1),axis=1)
    baseline=calculate_baseline(mean_int)
    corrected_data=mean_int-baseline
    noise=np.std(corrected_data) # valid for sparse blinking
    blink_ind=np.where(corrected_data > 10*noise)[0] #10 sigma massive event
    
    # add + and - 2 frames to make sure full blink captured 
    # (may change this based on frame rate)
    blink_ind=np.concatenate((blink_ind,blink_ind+2,blink_ind-2))
    blink_ind=list(set(blink_ind))
    
    return blink_ind

def check_contours(contours,frame):
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 500,500)
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(frame, contours, -1, (128,255,0), 1)
    cv2.imshow("Frame",frame)
    for n,c in enumerate(contours):
        area=cv2.contourArea(c)
        print('Countour # {}, Area: {}'.format(n,int(area)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def mask_artifact(image,dilate_mask=True,plot=False):
    frame=image.copy()
    ret, thresh = cv2.threshold(frame, params.ARTIFACT_LOW_VALUE,
                            params.ARTIFACT_HIGH_VALUE,cv2.THRESH_BINARY)
    # no invert, looking for white pixels
    contour_img,contours,hier = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    #check_contours(contours,frame)

    mask= np.ones(frame.shape,dtype="uint8") * 255

    # get largest area contour
    ind=np.argmax([cv2.contourArea(x) for x in contours]) #find contour with most points
    big_contour=contours[ind]
    cv2.drawContours(mask,[big_contour],-1,0,-1) #if 1 cont, put into list (filled contour quirk)

    # Mask multiple large contours
    # Problematic becuase, belt and headbar reflections can be at the edge of 
    # the pupil, masking these causes problems
    #for contour in contours:
    #    area= cv2.contourArea(contour)
    #    if area > 50:
    #        cv2.drawContours(mask,[contour],-1,0,-1)

    #grow the artifact mask a little bit
    if dilate_mask:
        kernel=np.ones((3,3),np.uint8)
        d=cv2.dilate(~mask,kernel,iterations=1)
        mask=~d

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # reblur image with masked artifact
    # TODO, make blur just around mask, rather than whole frame again
    frame = cv2.GaussianBlur(frame, params.BLUR_SIZE, 0)

    if plot:
        cv2.imshow('mask',mask)
        cv2.imshow('output',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return frame
    
def find_pupil(frame,return_images=False,verbose=True,debug=False):
    circle_outside_frame = True
    pupil_found=True
    contour_found=True
    area=0
    loop1=0
    loop2=0
    low_thresh=params.LOW_PIXEL_THRESH

    sub_frame=frame[params.UPPER_BOUND:params.LOWER_BOUND,
                params.LEFT_BOUND:params.RIGHT_BOUND]
    sub_frame = cv2.GaussianBlur(sub_frame.copy(), params.BLUR_SIZE, 0)

    masked_frame=mask_artifact(sub_frame)
    masked_frame=sub_frame
    # In this loop, lower threshold is decreased until at least one contour 
    # found and the area of the contour is bigger than min area. All the 
    # contours with the most x,y coordinates is kept (WARNING: may not be pupil,
    # but usually is)
    while area < params.MIN_AREA:
        if loop1 >0:
            low_thresh+=1
            if verbose:
                print('NO CONTOUR - Loop: {}, Low Tresh: {}'\
                    .format(loop2,low_thresh))
        
        ret, thresh = cv2.threshold(masked_frame.copy(), low_thresh,
                                    params.HIGH_PIXEL_VALUE,cv2.THRESH_BINARY)
        # invert for findContours          
        _ ,contours,hier = cv2.findContours(~thresh.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

        # get contour with largest area (presumably pupil)
        if len(contours) > 0:
            ind=np.argmax([cv2.contourArea(x) for x in contours]) 
            big_contour=contours[ind]
            area=cv2.contourArea(big_contour)
        loop1+=1

        # can't find pupil, break loop, save output values as 0
        if loop1 > 25:
            pupil_found=False
            contour_found=False
            radius=0
            area=0
            center=(0,0)
            if debug:
                #for i,cnt in enumerate(contours):
                #    print('{} - Area: {}'.format(i,cv2.contourArea(cnt)))
                check_contours(contours,thresh.copy())
            break

    if contour_found:
        # now make a circle around the largest contour found
        (x,y),radius = cv2.minEnclosingCircle(big_contour)
        center = (int(x),int(y))
        radius = int(radius)
        
        # Check if circle is outside the frame, which indicates it found something
        # too big, that is likely not the pupil
        # Note: this only looks for negatives (ie top and left of frame), should be 
        # the entire frame
        extremes=[center[0]+radius,center[0]-radius,
                center[1]+radius,center[1]-radius]
        circle_outside_frame=any(n < 0 for n in extremes) 

        if circle_outside_frame:
            pupil_found=False
            if debug:
                #for i,cnt in enumerate(contours):
                #    print('{} - Area: {}'.format(i,cv2.contourArea(cnt)))
                check_contours(contours,thresh.copy())
        else:
            pupil_found=True

    # center was calculated for sub_frame above, add bounds to get center for
    # full frame
    center=(center[0]+params.LEFT_BOUND,center[1]+params.UPPER_BOUND)

    # Note: if pupil not found, keep the nonesense value from current the 
    # evaluation. These indices are saved and manually corrected later.
    if return_images:
        # draw the contours on sub_frame
        contour_img=cv2.cvtColor(thresh.copy(),cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_img, contours, -1, (128,255,0), 1)
        # draw the detected circle on sub_frame
        c_thresh=cv2.cvtColor(thresh.copy(),cv2.COLOR_GRAY2RGB)
        circle_img=cv2.circle(c_thresh,center,radius,(255,0,0),1)
        # draw the detected circle on original full frame
        full_circle=add_circle_to_frame(frame,center,radius)
        
        image_dict={'frame': sub_frame, 'thresh':thresh,'contour':contour_img,
                    'circle':circle_img,'full_circle':full_circle,
                    'mask':masked_frame}
        return pupil_found,radius,center,area,image_dict
    else:
        return pupil_found,radius,center,area

def add_circle_to_frame(frame,center,radius,pupil_dict={}):
    # if no dict passed, assume the global params
    # pupil_dict passed in usually when making a movie
    #if not pupil_dict:
    #    full_center=(center[0]+params.LEFT_BOUND,center[1]+params.UPPER_BOUND)
    #else:
    #    P=pupil_dict.get('params')
    #    full_center=(int(center[0])+P['LEFT_BOUND'],int(center[1])+P['UPPER_BOUND'])
    cf=cv2.cvtColor(frame.copy(),cv2.COLOR_GRAY2RGB)
    full_circle=cv2.circle(cf,center,int(radius),(255,0,0),1)

    return full_circle
   
def process_image_stack(images,initial_center,debug=False):
    radii_auto=[]
    centers_auto=[]
    bad_frames=[]
    blinks=[]
    shifts=[]
    areas_auto=[]
    last_center=initial_center
    pupil_dict={
}    
    # detect blinks, return indices
    blinks=detect_blinks(images)

    for n,frame in enumerate(images):
        print 'FRAME - {}'.format(n) 
        if n not in blinks: 
            pupil_found,radius,center,area=find_pupil(frame.copy(),return_images=False)
            if not pupil_found:
                bad_frames.append(n)
                #shifts.append(0) # no pupil,

                if debug:
                    show_test_frame(frame.copy(),n)
            else:
                #check for large shifts
                center_shift = math.hypot(center[0] - last_center[0],
                                center[1] - last_center[1])
                #import pdb
                
                #pdb.set_trace()

                shifts.append(center_shift)
                last_center=center
                
                #print("CENTER: {}".format(center_shift))
                if  center_shift < params.MAX_CENTER_SHIFT:
                    radii_auto.append(radius)
                    areas_auto.append(area)
                    centers_auto.append(center)
                else:
                    # why append 0's? these will be manually corrected
                    radii_auto.append(0)
                    centers_auto.append((0,0))
                    bad_frames.append(n)
                    areas_auto.append(0)          
        else:
            # blink detected, fill with nans
            radii_auto.append(np.nan)
            centers_auto.append((np.nan,np.nan))
            shifts.append(np.nan)
            areas_auto.append(np.nan)

    # will be filled post manual corrected
    # TODO: auto is list, while radii etc are array, do I care?
    radii=np.asarray(radii_auto)
    centers=np.asarray(centers_auto)
    areas=np.asarray(areas_auto)
    bad_frames=np.asarray(bad_frames)


    pupil_dict={'radii':radii,
                'radii_auto':radii_auto,
                'centers':centers,
                'centers_auto':centers_auto,
                'areas':areas,
                'areas_auto':areas_auto,
                'params':vars(params), # save class as dict, maybe not elegant
                'blinks':blinks,
                'bad_frames': bad_frames,
                'shifts': shifts,
                'manual_corrected': False,
                'frame_shape':images.shape[:1]}

    return pupil_dict

def process_single_frame(image,frame_index,plot=True):

    pupil_found,r,c,a,image_dict=find_pupil(image.copy(),
                                return_images=True,debug=False)
    if plot:
        fig, ax = plt.subplots(2, 3,figsize=(8,6))
        fig.suptitle("Frame: {}, Pupil Found: {}, Area: {}, Radius: {}".\
            format(frame_index,pupil_found,a,r), fontsize=16)
        ax[0, 0].imshow(image_dict['frame'],cmap='gray')
        ax[0, 0].set_title('Frame')
        ax[0, 2].imshow(image_dict['thresh'],cmap='gray')
        ax[0, 2].set_title('Threshold')
        ax[0, 1].imshow(image_dict['mask'],cmap='gray')
        ax[0, 1].set_title('Artifact')
        ax[1, 0].imshow(image_dict['contour'],cmap='gray')
        ax[1, 0].set_title('Contour')
        ax[1, 1].imshow(image_dict['circle'])
        ax[1, 1].set_title('Circle')

        ax_flat=list(itertools.chain(*ax))
        plt.setp([a.get_xticklabels() for a in ax_flat], visible=False)
        plt.setp([a.get_yticklabels() for a in ax_flat], visible=False)
        plt.show()

def show_test_frame(image,frame_index,plot=True):
    # plot test frame without finding pupil algo
    frame=image.copy()
    sub_frame=frame[params.UPPER_BOUND:params.LOWER_BOUND,
                        params.LEFT_BOUND:params.RIGHT_BOUND]
    sub_frame = cv2.GaussianBlur(sub_frame.copy(), params.BLUR_SIZE, 0)
    masked=mask_artifact(sub_frame.copy())
    ret, thresh = cv2.threshold(masked.copy(), params.LOW_PIXEL_THRESH,
                                params.HIGH_PIXEL_VALUE,cv2.THRESH_BINARY)
    #invert for contour
    contour_img,contours,hier = cv2.findContours(~thresh.copy(),cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if plot:
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Frame: {}".format(frame_index), fontsize=16)
        axs[0,0].imshow(frame,cmap='gray')
        axs[0,0].set_title('Full Frame')
        axs[0,1].imshow(sub_frame,cmap='gray')
        axs[0,1].set_title('Sub Frame')
        axs[1,0].imshow(thresh,cmap='gray')
        axs[1,0].set_title('Thresholded')
        axs[1,1].imshow(contour_img,cmap='gray')
        axs[1,1].set_title('Countours: {}'.format(len(contours)))

        plt.show()

def zproject_stack(images):
    mean_img=np.mean(images,axis=0)
    min_img=np.min(images,axis=0)
    fig, axs = plt.subplots(1, 2)
    #ax.plot(1,1)
    #plt.show()
    axs[0].imshow(mean_img,cmap='gray')
    axs[1].imshow(min_img,cmap='gray')
    plt.show()

def detection_movie_old(images,indices,save_dir='C:\\Users\\Matt\\Desktop\\',
                    filename='temp.tif'):
    # TODO: make option for function to accept already caclulated radius and center
     with TiffWriter(os.path.join(save_dir,filename), bigtiff=True) as tifw:
         for index in indices:
             print index
             p,r,c,a,image_dict=find_pupil(images[index],return_images=True)

             tifw.save(image_dict['full_circle'])

def detection_movie(pupil_dict={},indices=[],save_dir='C:\\Users\\Matt\\Desktop\\',
                    filename='temp.tif'):
    # TODO: make option for function to accept already caclulated radius and center
    image_stack= tif.imread(pupil_dict['tiff_path'])

    with TiffWriter(os.path.join(save_dir,filename), bigtiff=True) as tifw:
     for index in indices:
        if index not in pupil_dict['blinks']:
             print index
             center=pupil_dict['centers'][index]
             radius=pupil_dict['radii'][index]
             full_circle=add_circle_to_frame(image_stack[index],center,radius,pupil_dict)

             tifw.save(full_circle)

def save_pupil_data(pupil_dict,overwrite=False):
    # TODO: check for overwrite_flag
    if overwrite:
        print('Overwrite Not Implemented')
    save_path=pupil_dict['tiff_path'].replace('.tif','.pkl')
    pickle.dump(pupil_dict, open(save_path, 'wb'))
    return

def load_pupil_data(pupil_pkl_file):
    try:
        with open(pupil_pkl_file, "rb") as f:
            pupil_dict = pickle.load(f)
    except (IOError, pickle.UnpicklingError):
        return False
    return pupil_dict


def manually_correct_pupil(images,pupil_dict,frames=[]):
    # see - https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    # passing data in callback: https://github.com/nelpy/nelpy/blob/master/nelpy/homography.py
    global points,drawing,line_end

    radii_manual=[]
    centers_manual=[]
    areas_manual=[]

    if not frames:
        frames=pupil_dict['bad_frames']
   
    for n,index in enumerate(frames):
        points =[]
        line_end=[]
        image = images[index]
        image=cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
        
        clone = image.copy()
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800,800)
        
        cv2.setMouseCallback("image", draw_line,image)
        while True:
            if not drawing:
                cv2.imshow("image", image)
            elif drawing and line_end:
                # redraws line while mouse is held down.
                line_copy=image.copy()
                cv2.line(line_copy, points[0], line_end[0], (255,0,0), 1)
                cv2.imshow('image',line_copy)
                
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            # seems buggy 05232018
            if key == ord("r"):
            	image = clone.copy()
                points =[]
            # press 'c' to continue
            elif key == ord("c"):
            	break
        if len(points) == 2:
            diameter = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
            radius=diameter/2
            center =((points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2)             	
        
            radii_manual.append(radius)
            centers_manual.append(center)
            
            area=3.14*radius*radius
            areas_manual.append(area)
            
            print('Frame: {}, Center: {}, Radius: {}'.format(n,center,radius))

            cv2.destroyAllWindows()

    bad_frames = pupil_dict['bad_frames']
    pupil_dict['radii'][bad_frames]=radii_manual
    pupil_dict['centers'][bad_frames]=centers_manual
    pupil_dict['areas'][bad_frames]=areas_manual
    pupil_dict['manual_corrected']=True

    return pupil_dict

# global vars needed for callback
points = []
drawing = False
line_end=[]

def draw_line(event, x , y, ignored, image):
    global points, drawing, line_end
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x,y))
        drawing = False
        '''
        try:
            cv2.line(image, points[0], points[1], (255,0,0), 1)
            cv2.imshow("image", image)
        except IndexError
            continue
        '''

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        line_end=[(x, y)]
        
    
    
def main(argv):
    argParser = argparse.ArgumentParser()
    #argParser.add_argument(
    #    '-o', '--overwrite', action='store_true',
    #    help='overwirte existing .pkl file')
    #argParser.add_argument(
    #    '-d', '--directory', type=str,
    #    help='process all tif files beneath this directory')
    argParser.add_argument(
        '-f', '--file', action='store', type=str,
        help='process specifically this file')
    argParser.add_argument(
        '-s', '--single', action='store_true',
        help='process single frame')
    argParser.add_argument(
        '-D', '--debug', action='store_true',
        help='debug during process stack, will plot frames where pupil is not found\
            Close the plot to continue processing.')
    argParser.add_argument(
        '-n', '--no_manual', action='store_true',
        help='skip manual pupil fixing step')
    argParser.add_argument(
        '-t', '--test_mode', action='store_true',
        help='plot single frame, adjust params, and test again')
    argParser.add_argument(
        '-z', '--zproject', action='store_true',
        help='zproject stack, look at min and mean, useful for pupil bounding')
    argParser.add_argument(
        '-p', '--process_stack', action='store_true',
        help='process the entire stack')
    argParser.add_argument(
        '-m', '--movie', action='store_true',
        help='')
    argParser.add_argument(
        "-F","--frame", action="store", type=int, nargs='+',
        help="select frame or frame (start and end indices) to process")
    argParser.add_argument(
        "-P", "--plot", action='store_true',
        help="load pkl and plot")

    args = argParser.parse_args(argv)    
    
    #%% STEP 1
    initial_center=(95,95)

    # TODO: make args.frame check for process stack if two given
    # make default for whole stack
    #if args.process_stack:
    #    try len(args.frame)>1:
    #if not args.frame and not args.process_stack:
    if not args.frame:
        frame_index=[0]
        print('No frame index given, assuming 0')
    else:
        frame_index=args.frame

    #TODO: try
    tiff_path=args.file
    #tiff_path='C:\\Users\\Matt\\Desktop\\CREB9\\mjd535_20180614121218.tif'
    #tiff_path='/media/bzlab/pupil/CREB9/mjd535_20180614121218.tif'
    
    if args.single:
        image= tif.imread(tiff_path,key=frame_index[0])
        process_single_frame(image,frame_index=frame_index[0],plot=True)
    if args.test_mode:
        image = tif.imread(tiff_path,key=frame_index[0])
        show_test_frame(image,frame_index=frame_index[0])
    if args.zproject:
        images = tif.imread(tiff_path)
        zproject_stack(images)
    if args.process_stack:
        if len(frame_index) == 3:
            # 3rd index is incremental
            image_stack= tif.imread(tiff_path,key=slice(frame_index[0],
                                    frame_index[1],frame_index[2]))
        elif len(frame_index) == 2:
            image_stack= tif.imread(tiff_path,key=slice(frame_index[0],
                                    frame_index[1]))
        else:
            image_stack= tif.imread(tiff_path)
        print('Loaded image stack: {}'.format(image_stack.shape))

        pupil_dict =process_image_stack(image_stack,
                    initial_center=initial_center,
                    debug=args.debug)
        print('Finished processing')
        print('# bad frames: {}'.format(len(pupil_dict['bad_frames'])))
        
        pupil_dict['tiff_path'] = tiff_path

        # manually correct shifted frames
        if not args.no_manual:
            pupil_dict = manually_correct_pupil(image_stack,pupil_dict)
        # save
        save_pupil_data(pupil_dict,overwrite=False)

    
    if args.plot:
        #TODO check for pkl
        pkl_file=args.file
        pd = load_pupil_data(pkl_file)
        
        #TODO: MAKE PLOT auto save to PDF
        fig, ax = plt.subplots(4, 1)

        center_x=[c[0] for c in pd['centers']]
        ax[0].plot(pd['radii'])
        ax[0].set_title('Radius')
        ax[1].plot(center_x)
        ax[1].set_title('Center X')
        ax[2].plot(pd['areas'])
        ax[2].set_title('Area')
        ax[3].plot(pd['shifts'])
        ax[3].set_title('shifts')

        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(np.arange(0,len(pd['radii'])),pd['shifts'])
        ax[0].set_title('Radius vs between frames shifts')
        ax[1].scatter(np.arange(1,len(pd['radii'])),np.diff(pd['areas_auto']))
        ax[1].set_title('Radius vs area (pre-manual)')
        plt.show()

    if args.movie:
        pkl_file='C:\\Users\\Matt\\Desktop\\mjd535_20180614121218.pkl'
        pd = load_pupil_data(pkl_file)

        detection_movie(pupil_dict=pd,indices=np.arange(0,100,1),
                        filename='505-movie.tif')

if __name__ == '__main__':
    main(sys.argv[1:])

