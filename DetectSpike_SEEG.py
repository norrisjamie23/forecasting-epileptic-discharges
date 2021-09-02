from scipy.signal import butter, filtfilt
import math
import numpy as np
###########################################################################
# CodeID:           SPKDT v1.0.4
# Author:           Danielle T. Barkmeier
# Modified by:      Rajeev Yadav
# Email:            rajeevyadav@gmail.com
# Dated:            September, 11th 2011
# Rev. Update:
#                   1. Fixed bug in filter specification
# Modified and debugged by Brian Lundstrom, 1/2016
# Converted to Python and modified by Jamie Norris, 5/2021
###########################################################################

def DetectSpikes(Data, Fs, BlockSize=1, SCALE=70, STDCoeff=4, DetThresholds=[7, 7, 600, 10, 10], FilterSpec=[20, 50, 1, 35], TroughSearch=40):
# This function detects spikes in the EEG. The input parameters to the function are
# (1) Data = EEG signal of N x M dimension where M = data length and N = # of
# channels, with negative potentials as positive (viewing convention). 
# (2) Fs = samping rate
# (3) BlockSize = length of data block to be processed in minutes
# (default = 1 minute block).
# (4) SCALE = scale the EEG amplitude in percentage. Default is 70#.
# (5) STDCoeff = distance from the mean (measure of spread across the mean
# default = 4 times the standard deviation from the mean).
# (6) DetThresholds = specify the detection thresholds which contains
# thresholds for the left half-wave slope, right half-wave slope, total
# amplitude of the spike (left half-wave + right half-wave), duration of
# left half-wave and right half-wave in ms. Default values for the  
# DetThresholds = [7; 7; 600; 10 ; 10];
# (7) FilterSpec contains the filter specification. The cut-off frequency
# for the first high-pass and low-pass filter, followed by cut-off
# frequency for the second high-pass and low-pass filter. Filter order is
# kept fixed.
# (8) TroughSearch is distance in ms to search for a trough on each side of
# a detected peak. Default width is 40 ms.
# The function returns returns time-instance of detected spike events (SpikeIndex),
# detection channel (ChanId), and spike feature (SpikeFV). The SpikeFV is of 
# dimension P x 8, where P represents numbers of spikes detected in the data: 
# (1)Index of peak 
# (2)Value at peak of spike, 
# (3)Left half-wave amplitude,
# (4)Left half-wave duration, 
# (5)Left half-wave slope, 
# (6)Right half-wave amplitude, 
# (7)Right half-wave duration, and
# (8)Right half-wave slope.
# ############################

# Usage: [SpikeIndex, ChanId, SpikeFV] = mDetectSpike(Data, Fs, BlockSize,
# SCALE, STDCoeff, DetThresholds, FilterSpec, TroughSearch);

    if len(DetThresholds) != 5:
        print("Using default detection thresholds")
        LS = 7
        RS = 7
        TAMP = 600
        LD = 10
        RD = 10
    else:
        LS = DetThresholds[0]
        RS = DetThresholds[1]
        TAMP = DetThresholds[2]
        LD = DetThresholds[3]
        RD = DetThresholds[4]
    if len(FilterSpec) != 4:
        print("Using default filter specification")
        FilterSpec = [20, 50, 1, 35]

    # Filter specification
    bh1, ah1 = butter(2, FilterSpec[0] / (Fs / 2), 'highpass')
    bl1, al1 = butter(4, FilterSpec[1] / (Fs / 2), 'lowpass')
    bh2, ah2 = butter(2, FilterSpec[2] / (Fs / 2), 'highpass')
    bl2, al2 = butter(4, FilterSpec[3] / (Fs / 2), 'lowpass')

    # Examine data and initialize parameters for processing
    N, M = Data.shape    # where M = data size, N = number of channels
    NumSecs = M / Fs    # Get file length in seconds
    ChanGroups = list(range(N))    # Channel group
    # Initialization
    AllSpikes = []
    SpikeIndex = [] 
    ChanId = []
    ChanPeaks = []
    SpikeFV = []
    c = (1 / (1000 / np.round(Fs)))     # convert ms to # of data points
    c_times_25 = np.round(25 * c).astype(int)
    
    # Requires blocksize (min) as input parameter
    Blocks = math.floor(NumSecs / (BlockSize * 60))    # exclude partial data at end
    BlockIdxSize = math.ceil(BlockSize * 60 * Fs)    # number of data points per Block

    for CurrentBlock in range(Blocks):

        # Get data for correct time blocks
        EEG = Data[:, CurrentBlock * BlockIdxSize : (CurrentBlock + 1) * BlockIdxSize].copy()

        # Detect artifact channel
        d1s = np.mean(np.absolute(np.diff(EEG, axis=1)), axis=1)
        tmp = d1s

        # Screen out any values more than X the median slope
        # tmp[np.where(tmp > np.median(tmp) * 2)] = None TODO fix this and re-enable

        # Detect artifact channels by those whose average slope are greater
        # than 10 standard deviations outside the mean of this screened data
        artifactChans = np.where(d1s > 10 * np.std(tmp) + np.mean(tmp))[0]

        # Remove them from consideration
        BlockChanNums = list(set(ChanGroups) - set(artifactChans))

        # Filter data in each channel for each block
        fEEG = np.empty_like(EEG)
        EEG[BlockChanNums] = -EEG[BlockChanNums]    # flip data so that negative is downwards
        fEEG[BlockChanNums] = filtfilt(bh1, ah1, EEG[BlockChanNums]) 
        fEEG[BlockChanNums] = filtfilt(bl1, al1, fEEG[BlockChanNums])    # Narrow bandpass
        # Filter original data
        EEG[BlockChanNums] = filtfilt(bh2, ah2, EEG[BlockChanNums])    # High pass filter
        EEG[BlockChanNums] = filtfilt(bl2, al2, EEG[BlockChanNums])    # low pass --band passed but wider than fEEG

        # Scale channels
        # Default scaling = 70, lower scale can increase effect of background
        # With high scale, DetThresholds are irrelevant
        # With 70 and [7 7 600...], thresholds are .1 .1 8.5 
        # as normalized by the median mean abs amp of the EEG block
        EffScale = SCALE/np.median(np.mean(np.absolute(EEG[ChanGroups])))
        EEG[ChanGroups] = EEG[ChanGroups]*EffScale

        # Detect spikes
        # default STDCoeff = 4 % Z-score

        for Chan in BlockChanNums:
            thresh = (-np.mean(np.absolute(fEEG[Chan])) - STDCoeff * np.std(np.absolute(fEEG[Chan])))    # -(mean+std)
            peaks = np.where(fEEG[Chan, math.ceil(500 * c) + 1 : - 500] < thresh)[0]    # ignore edges by XX msec

            peaks += math.ceil(500 * c) + 1
            
            # If current channel has no peaks
            if len(peaks) == 0:
                continue
                                    
            # Creating these ranges allows us to extract multiple slices at the same time
            ranges = [np.arange(peak - c_times_25, peak + c_times_25) for peak in peaks]
            
            # Narrow peaks down to a single time point
            newPeakVs = fEEG[Chan, [ranges]][0].min(axis=1)
            
            # Get peak indices
            newPeakIs = np.argmin(fEEG[Chan, [ranges]][0], axis=1)
            newPeakIs += peaks - c_times_25 - 1
            
            # Keep unique peaks only
            _, unique_indices = np.unique(newPeakIs, return_index=True)
            newPeakVs = newPeakVs[unique_indices]
            newPeakIs = newPeakIs[unique_indices]
            
            ranges = [np.arange(newPeakI - math.ceil(20 * c), newPeakI + math.ceil(20 * c)) for newPeakI in newPeakIs]
            
            spikeVs = EEG[Chan, [ranges]][0].min(axis=1)
            spikeIs = np.argmin(EEG[Chan, [ranges]][0], axis=1)
            spikeIs += newPeakIs - math.ceil(20 * c) - 1
            
            ranges = [np.arange(spikeI - math.ceil(TroughSearch * c), spikeI) for spikeI in spikeIs]
            
            leftVs = EEG[Chan, [ranges]][0].max(axis=1)
            leftIs = np.argmax(EEG[Chan, [ranges]][0], axis=1)
            leftIs += spikeIs - math.ceil(TroughSearch * c) - 1
            
            ranges = [np.arange(spikeI, spikeI + math.ceil(TroughSearch * c)) for spikeI in spikeIs]
            
            rightVs = EEG[Chan, [ranges]][0].max(axis=1)
            rightIs = np.argmax(EEG[Chan, [ranges]][0], axis=1)
            rightIs += spikeIs - 1
            
            # add negative signs due to calculation with negative downward
            # Get amp, dur and slope for the left halfwave
            Lamps = - (spikeVs - leftVs)
            Ldurs = (spikeIs - leftIs) / c
            Lslopes = Lamps / Ldurs

            # Get amp, dur and slope for the right halfwave
            Ramps = - (spikeVs - rightVs)
            Rdurs = (rightIs - spikeIs) / c
            Rslopes = Ramps / Rdurs
            
            # Which spikes have valid slopes?
            valid_slope = np.logical_and(Lslopes > LS, Rslopes > RS)
            
            # Which spikes have valid amplitudes and are more than 5ms after previous spike?
            valid_amps_diff = np.logical_and(Lamps + Ramps > TAMP, np.insert(np.diff(spikeIs) > 5 * c, 0, True))
            
            # Which spikes are sufficiently long in duration?
            valid_duration = np.logical_and(Ldurs > LD, Rdurs > RD)
            
            # Combine slope and amplitude masks
            valid_slope_amps = np.logical_and(valid_slope, valid_amps_diff)
            
            # Combine all together to get final mask
            valid = np.logical_and(valid_slope_amps, valid_duration)
            
            # Array containing information on all valid peaks
            ChanPeaks = np.stack((spikeIs[valid], spikeVs[valid], Lamps[valid], 
                                  Ldurs[valid], Lslopes[valid], Ramps[valid], 
                                  Rdurs[valid], Rslopes[valid]), axis=1)
            
            # Calculate spike indices for current block/channel as an array and add to list across all
            SpikeIndex.append(ChanPeaks[:, 0] + CurrentBlock * BlockSize * 60 * np.round(Fs).astype(int))
            
            # Add current channel to ChanId's, once per spike
            ChanId.extend([Chan] * valid.sum())
            
            SpikeFV.append(ChanPeaks)
                             
    return np.concatenate(SpikeIndex, axis=0).astype(np.int32), np.array(ChanId).astype(np.int32), np.concatenate(SpikeFV, axis=0)