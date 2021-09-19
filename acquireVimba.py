
#good info on ssaving : https://github.com/morefigs/pymba/issues/13
# http://anki.xyz/installing-pymba-on-windows/
# https://github.com/morefigs/pymba/issues/13

# 30hz, 120 buffer seemed to work
    # 
# 60hz, 120 had some double images, maybe buffer bigger

'''
Need:
1)> opencv 3.2
2) tifffile (conda install tifffile -c conda-forge)
3)pymba
4) Vimba  SDK

SelectROI bug fix
conda install opencv-contrib-python
conda install -c menpo opencv3 
opencv-contrib-python     3.3.1.11 
opencv3                   3.1.0
'''
import os.path
from datetime import datetime
import argparse
import time

import numpy as np
import cv2
import pymba
from pymba import *
from tifffile import TiffWriter

# Global
framei=0

def set_freerun_parameters(camera0):
    # standards for pupil recording during treadmill running
    camera0.PixelFormat = 'Mono8'
    camera0.AcquisitionFrameRateAbs = 60

    camera0.TriggerSelector ='AcquisitionStart'
    camera0.TriggerMode ='Off'

    camera0.TriggerSelector ='AcquisitionEnd'
    camera0.TriggerMode ='Off'

    camera0.TriggerSelector ='FrameStart'
    camera0.TriggerSource = 'Freerun'


def set_acquisition_parameters(camera0):
	# standards for pupil recording during treadmill running
    camera0.PixelFormat = 'Mono8'
    camera0.AcquisitionFrameRateAbs = 30
    
    camera0.TriggerSelector ='FrameStart'
    camera0.TriggerSource = 'FixedRate'
    camera0.TriggerMode ='Off'

    camera0.TriggerSelector ='AcquisitionStart'
    camera0.TriggerSource = 'Line1'
    camera0.TriggerMode ='On'
    camera0.TriggerActivation = 'LevelHigh'

    camera0.TriggerSelector ='AcquisitionEnd'
    camera0.TriggerSource = 'Line1'
    camera0.TriggerMode ='On'
    camera0.TriggerActivation = 'LevelLow'


def aquire_data(camera0,save_dir='C:\\Users\\bzlab\\Desktop\\',mouse=None):
    start_time=datetime.now().strftime('%Y%m%d%H%M%S')
    if mouse is None:
        mouse ='mouse'
    fn=mouse+ '_' + start_time + '.tif'
    filename=os.path.join(save_dir,fn)
    tiff_logger= TiffWriter(filename, bigtiff=True)

    camera0.AcquisitionMode = 'Continuous'
    cv2.destroyAllWindows()
    
    global framei
    
    def frame_callback(frame):
       
        frame_data = frame.getBufferByteData()
        img = np.ndarray(buffer=frame_data,
                         dtype=np.uint8,
                         shape=(frame.height, frame.width))
        print img.mean()
        tiff_logger.save(img)

        global framei
        framei+=1
        
        frame.queueFrameCapture(frame_callback)

    n_vimba_frames = 10 # UNSURE ABOUT THIS VALUE, works for 30 fps
    frame_pool = [camera0.getFrame() for _ in xrange(n_vimba_frames)]
    for frame in frame_pool:
        frame.announceFrame() # now Vimba knows about the frame... also odd design choice
        frame.queueFrameCapture(frame_callback)

    camera0.startCapture()

    # Loop1 - pre acquistion
    # wait for external trigger to call aquisition start
    # can break with ctr+c
    now=time.time()
    print("************* READY TO AQUIRE *************")
    try:
        while framei==0:
            #print(int(round(time.time()-now)))
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # Loop2 - acquistion
    # wait for frame number to stop growing
    # can break with ctr+c
    try:
        while 1:
            last_size=framei
            time.sleep(2)
            now_size=framei
            if last_size == now_size:
                break
    except KeyboardInterrupt:
        pass

    # end acquisition sequence 
    time.sleep(0.2)
    camera0.flushCaptureQueue()
    camera0.endCapture()
    camera0.revokeAllFrames()
    tiff_logger.close()


def get_single_frame(camera0,display_image=True):
    camera0.AcquisitionMode = 'SingleFrame'

    frame0 = camera0.getFrame() 
    frame0.announceFrame()

     # capture a camera image
    camera0.startCapture()
    frame0.queueFrameCapture()
    camera0.runFeatureCommand('AcquisitionStart')
    camera0.runFeatureCommand('AcquisitionStop')
    frame0.waitFrameCapture()
    

    # get image data... 
    imgData = np.ndarray(buffer = frame0.getBufferByteData(),
                        dtype = np.uint8,
                        shape = (frame0.height,frame0.width, 1))

    if display_image:
        cv2.imshow("SingleFrame",imgData)
        cv2.waitKey(0)

    #clean-up
    camera0.endCapture()
    camera0.revokeAllFrames()

    return imgData

def run_continuous(camera0):
    cv2.destroyAllWindows()
    camera0.AcquisitionMode = 'Continuous'

    frame0 = camera0.getFrame() 
    frame0.announceFrame()

    camera0.startCapture()
    frame0.queueFrameCapture()
    camera0.runFeatureCommand('AcquisitionStart')
    framei = 0
    now = time.time()
    while 1:
        # The 1000 may be adjusted to your needs
        # Common source for errors
        frame0.waitFrameCapture(1000)
        frame0.queueFrameCapture()

        imgData = np.ndarray(buffer = frame0.getBufferByteData(),
                                       dtype = np.uint8,
                                       shape = (frame0.height,frame0.width,1))                                    
        framei += 1
        # Only show every nth frame
        key = cv2.waitKey(1)
        if framei % 6 == 0:
            cv2.putText(imgData,"%d fps" % (framei/(time.time()-now)), (10,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0))   
            cv2.imshow("Movie - Press 'q' to quit",imgData)
            key = cv2.waitKey(1)
        
        if key == ord("q"):
            break
        
    # clean up after capture
    frame0.waitFrameCapture(1000)
    camera0.runFeatureCommand('AcquisitionStop')
    camera0.endCapture()
    camera0.revokeAllFrames()
        
    print framei,"-", time.time()-now,"s - ", framei/(time.time()-now)

    cv2.destroyAllWindows()


def set_ROI(imgData):
    #invert image here to see the cross hairs better
    coords=cv2.selectROI('SingleFrame',cv2.bitwise_not(imgData),fromCenter=False)
    print(coords)
    
    camera0.OffsetX = int(coords[0])
    camera0.OffsetY = int(coords[1])
    camera0.Width = int(coords[2])
    camera0.Height = int(coords[3])
    # TODO- check if nothing selected, default to max perahps
    cv2.destroyAllWindows()
    get_single_frame(camera0) # works ok if I dont capture frames

    return

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-e", "--exposure", type=int,
        help="set camera exposure in microseconds")
    argParser.add_argument(
        "-r", "--roi", action="store_true",
        help="select ROI for image capture")
    argParser.add_argument(
        "-v", "--view_reset", action="store_true",
        help="reset fov to full height and width")
    argParser.add_argument(
        "-c", "--continuous", action="store_true",
        help="stream continuous movie")
    argParser.add_argument(
        "-a", "--aquire", action="store_true",
        help="stream continuous movie")
    argParser.add_argument(
        "-m", "--mouse_name", type=str,
        help="mouse name to save")

    args = argParser.parse_args()


    # start camera
    with Vimba() as vimba:
        print('vimba version = {}'.format(vimba.getVersion()))
        system = vimba.getSystem()

         # list available cameras (after enabling discovery for GigE cameras)
        if system.GeVTLIsPresent:
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(0.2)
        cameraIds = vimba.getCameraIds()
        for cameraId in cameraIds:
            print('camera = {}'.format(cameraId))

        # get & open camera
        camera0 = vimba.getCamera(cameraIds[0])
        camera0.openCamera()

        # Features
        cameraFeatureNames = camera0.getFeatureNames()
        #for name in cameraFeatureNames:
        #    print(name)

        # set exposure
        if args.exposure:
            camera0.ExposureTimeAbs = args.exposure
        else:
            f=1 
            #camera0.ExposureTimeAbs = 800
        print ('exposure = {}'.format(camera0.ExposureTimeAbs))

        if args.view_reset:
            camera0.OffsetX = 0
            camera0.OffsetY = 0
            camera0.Width = camera0.WidthMax
            camera0.Height = camera0.HeightMax

        set_freerun_parameters(camera0)

        imgData=get_single_frame(camera0,display_image=True)

        if args.continuous:
            run_continuous(camera0)
        
        if args.roi:
            set_ROI(imgData)

        if args.aquire:
            set_acquisition_parameters(camera0)
            aquire_data(camera0)

            if framei is not 0:
                #print("Aquired frames, now saving....")
                #save_list_of_arrays(time_series,mouse=args.mouse_name)
                print("Save complete!")
                print("Frames = " + str(framei))
            else:
                print('No Images Acquired :(')

        # Plotting
        #plt.imshow(np.squeeze(imgData))
        #plt.show()

        # clean up after capture
        cv2.destroyAllWindows()
        camera0.closeCamera()
        vimba.shutdown()
        