from lab.misc.auto_helpers import locate
import lab.analysis.behavior_analysis as ba

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy.signal as signal
from scipy.signal import argrelextrema
import cPickle as pickle
from copy import copy, deepcopy
from ..classes.new_interval import Interval
from lab.analysis import new_intervals as inter



def load_pupil(pupil_path):
    #path = self.pupilFilePath()
    path = pupil_path
    try:
        with open(path, 'rb') as fp:
            ft = pickle.load(fp)
    except IOError:
        return None

    else:
        return ft




def detect_pupil_phases(pupil_dict,order=4,sample_freq=30.0,cutoff_freq=0.5,
            MIN_EPOCH=30,MIN_RATE=1,GAP_EPOCH=15,seconds=True,
            plot_extrema=False,plot_phases=False,norm_minmax=True):
    """
    return pupil phases in seconds

    MIN_EPOCH frames rn
    MIN_RATE (notimplemented minmum value of mean rate of change of pupil epoch (reimer et al has 10um/s)
    GAP_EPOCH (not implemented) in frames, smallest gap allow within a rise or fall
    seconds reture in seconds

    """
    
    N = order #order
    fs = sample_freq #freq sample, make sure float
    fc = cutoff_freq #freq cutoff, make float

    #interpolate NaNs (usually blinks) before filtering
    radii_int=pd.Series(pupil_dict['radii']).interpolate().get_values().tolist()

    B,A = signal.butter(N,fc/(fs/2),output='ba')
    #radius_filt=signal.filtfilt(B,A,radii[1:4600])
    radius_filt=signal.filtfilt(B,A,radii_int)

    if norm_minmax:
        radius_filt= (radius_filt - np.min(radius_filt))/\
            (np.max(radius_filt)-np.min(radius_filt))
        radii_int= (radii_int - np.min(radii_int))/\
            (np.max(radii_int)-np.min(radii_int))

    rmax=argrelextrema(radius_filt,np.greater)[0]
    rmin=argrelextrema(radius_filt,np.less)[0]

    #min max
    if plot_extrema:
        fig, ax = plt.subplots(1, 1)
        ax.plot(radii_int,'b-')
        ax.plot(radius_filt,'r-')   
        ax.scatter(rmax,radius_filt[rmax],s=200,marker='*')
        ax.scatter(rmin,radius_filt[rmin],s=200,marker='o')


    #always make sure rmin is first
    if rmin[0] > rmax[0]:
        rmax=rmax[1:]
    # trim rmin by one if that array is longer
    if len(rmin) > len(rmax):
        rmin=rmin[:-1]
    # trim rmax (if needed), make rmin rmax equal
    if len(rmax) > len(rmin):
        rmin=rmax[:-1]

    # rising phase interval
    rise=rmax-rmin
    rise_ind=np.argwhere(rise>MIN_EPOCH)
    rise_start=rmin[rise_ind]
    rise_end=rise_start+rise[rise_ind]+1 #added one to get frame b/t rmax end rmin start

    # falling phase interval
    fall=rmin[1:]-rmax[:-1]
    fall_ind=np.argwhere(fall>MIN_EPOCH)
    fall_start=rmax[fall_ind]
    fall_end=fall_start+fall[fall_ind]

    # can add a term to look for small gaps (<.5s), remove the break
    # first dataset lowest gap ~ 25 frames
    # NOT IMPLEMENTED
    gaps=rise_start[1:]-rise_end[:-1]
    gap=np.argwhere(gaps<GAP_EPOCH)


    dilation=np.concatenate([rise_start,rise_end],axis=1)
    constriction=np.concatenate([fall_start,fall_end],axis=1)
    #full_cycle= np.concatenate([rise_start,fall_end],axis=1)
    #maybe implent full pupil cycle, start dilation, end constriction
    #will require fusing any consecutive events (e.g. dilat, then dilat)
    #cycle=np.concatenate([rise_start,rise_end],axis=1)

    # NOT DONE
    # get absolute mean slope for each event
    # may use this as a filtering criteria (like McCormick)
    d_mean_slope=[]
    c_mean_slope=[]
    for start, end in dilation:
        d_mean_slope.append(
            np.absolute(np.nanmean(np.diff(radius_filt[start:end]))))
    for start, end in constriction:
        c_mean_slope.append(
            np.absolute(np.nanmean(np.diff(radius_filt[start:end]))))



    if seconds:
        dilation=dilation/fs
        constriction=constriction/fs

    if plot_phases:
        fig, ax = plt.subplots(1, 1)
        ax.plot(radius_filt,'b-')
        for start,end in dilation:
            ax.plot(np.arange(start,end,1),
                    radius_filt[start:end],'r-')
        for start,end in restriction:
            ax.plot(np.arange(start,end,1),
                    radius_filt[start:end],'g-')
                    
    pupilDict={}
    pupilDict['dilation']=dilation
    pupilDict['d_slope']=d_mean_slope
    pupilDict['constriction']=constriction
    pupilDict['c_slope']=c_mean_slope
    pupilDict['all']=radius_filt


    # not sure if i need deepcopy
    return deepcopy(pupilDict)


def pupil_frames(pupilDict,imageSync=True,sampling_interval=None):
    '''Return pupil phases epochs in imaging frames'''
    phases=['dilation','constriction','all']
    # not sure if i need deepcopy
    dataDict=deepcopy(pupilDict)
    if imageSync:
        sampling_interval=1/15.024038461 # 
        #sampling_interval=1/7.8125 # 
        nFrames=11000 #HARD CODED
        #sampling_interval=self.parent.frame_period()
        #nFrames=self.parent.num_frames()
        
    if sampling_interval=='actual':
        sampling_interval=1/fs
    
    recordingDuration=720 #HARD CODED
    
    numberBehaviorFrames=int(recordingDuration/ sampling_interval)
    
    for phase in phases:
        out= np.zeros(numberBehaviorFrames, 'bool')
        if phase == 'all':
            # not sure if this is the best
            out = signal.resample(dataDict[phase], numberBehaviorFrames)
        else:
            for start,stop in dataDict[phase]:
                if np.isnan(start):
                    start=0
                if np.isnan(stop):
                    stop=recordingDuration
                start_frame = int(start/sampling_interval)
                stop_frame=int(stop/sampling_interval)
                out[start_frame:stop_frame] = True
            
        if imageSync and len(out) > nFrames:
            out = out[:nFrames]
        dataDict[phase]=out
    # not sure if i need deepcopy
    return deepcopy(dataDict)
    





def pupil_phase_recruitment(exptGrp, phase='dilation',roi_filter=None, 
                            label=None, dtype='phase_in_trans',activity_interval=None):
    """Finds events/data (e.g. transients) in a pupil phase (e.g. dilation)

    TODO: summary

    Parameters
    ----------
    exptGrp: lab.ExperimentGroup
    phase: str
        'dilation' or 'constriction'
    roi_filter: ???
    label: ???
    dtype: str
        'phase_in_trans': for nRois, how many transients were in the dilation or constriction phase
        'trans_in_phase': for nPhases (dilation or constriction), how many transients were in each
        individual phase event (pooling across all ROIs)
    activity_interval: str
        'running', 'stationary', or None

    Returns
    -------
    data_list: dataframe,
        'value' column contains calculation, with rows equal to nRois or nPhases
    pupilFrames:
        ???
    pupil_mask:
        ???
    pupil_phases:
        ???
    """
    pupil_path = '/media/bzlab/imaging/creb/pupil/mjd511/mjd511_20171226210807.pkl'

    pupilData=load_pupil(pupil_path)
    pupil_phases=detect_pupil_phases(pupilData)
    pupilFrames=pupil_frames(pupil_phases)
    
    data_list = []

    for expt in exptGrp:

        if activity_interval == 'stationary':
            # CHECK THIS FRAME PERIOD
            # couldn't figure out the best way to get overlapping intervals
            #pupil_mask=Interval.from_mask(pupilFrames[phase], sampling_interval=trial.frame_period(),
            #                    data={'trial': trial})
            #stationary_int=inter.stationary_intervals(exptGrp)

            stationary_intervals = ~np.array(expt.runningIntervals(
                                    returnBoolList=True))
            pupil_mask = pupilFrames[phase] & stationary_intervals
            pupil_mask = pupil_mask.T.squeeze()
        elif activity_interval == 'running':
            running_intervals = np.array(expt.runningIntervals(
                                    returnBoolList=True))
            pupil_mask = pupilFrames[phase] & running_intervals
            pupil_mask = pupil_mask.T.squeeze()
        else:
            pupil_mask=pupilFrames[phase]


        n_pupil_phases = len(pupil_mask)
        #n_ripples = len(expt.ripple_frames())
        rois = expt.rois(label=label, roi_filter=roi_filter)

        if dtype == 'phase_in_trans':
            trans = expt.transientsData(label=label, roi_filter=roi_filter)
            #ripples = expt.rippleIntervals(returnBoolList=True)
            for ri, roi_trans in enumerate(trans):
                in_pupil = 0
                for start in roi_trans['start_indices'][0]:
                    in_pupil += pupil_mask[start]

                data_list.append({'expt': expt,
                                  'roi': rois[ri],
                                  #'value': in_pupil / float(n_pupil_phases)})
                                  'value': in_pupil})

        if dtype == 'trans_in_phase':
            trans = expt.transientsData(label=label, roi_filter=roi_filter)
            #ripples = expt.rippleIntervals(returnBoolList=True)
            
            #convert the filtered (e.g. staionary/running) pupil mask back to Interval 
            pupil_intervals=Interval.from_mask(pupil_mask,
                sampling_interval=1/15.024038461,data={'expt': expt}) #HARD CODE

            for index, row in pupil_intervals.iterrows():
                pupil_sub_mask=np.zeros(len(pupil_mask))
                pupil_sub_mask[row['start']:row['stop']]=1

                in_pupil = 0
                duration_set=[]
                for ri, roi_trans in enumerate(trans):
                    duration=[]
                    for ti, start in enumerate(roi_trans['start_indices'][0]):
                        transient_counter=in_pupil #clunky?

                        in_pupil += pupil_sub_mask[start]
                        # transient counter logic seems kinda clunky
                        if in_pupil > transient_counter:
                            duration.append(roi_trans['durations_sec'][0][ti])
                            #duration.append(roi_trans['max_amplitudes'][0][ti])
                            #roi_trans['max_amplitudes']
                            #roi_trans['max_indices']
                    duration_set.extend(duration)



                data_list.append({'expt': expt,
                                  #'roi': rois[ri],
                                  #'value': in_pupil / float(n_pupil_phases)})
                                  'value': in_pupil,
                                  'durations': duration_set})


    return pd.DataFrame(data_list),pupilFrames[phase],pupil_mask,pupil_phases,pupilFrames['all']


