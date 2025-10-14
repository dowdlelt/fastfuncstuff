#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:16:41 2020

This module provides a set of functions to create non-randomized 
alternating sequences of events and add  realisitic components 
similar to the data obtained from a fMRI  experiment. It also 
provides tools to  calcuate optimality measures of the sequences.

The two main components are: expdesign class that can be used to 
design a timecourse of events, expanalyse class that can be used 
to calcuate its statistical efficiency.


@author: Soukhin Das (skndas@ucdavis.edu), Center for Mind and Brain, Davis, California
@author: Weigang Yi, Center for Mind and Brain, Davis, California
University of California, Davis
"""

import numpy as np
import fmrisim_modified as fmrisim
from tools._dghrf import _dghrf
import statsmodels.api as sm
from scipy.linalg import toeplitz
from importlib import reload
reload(fmrisim)

class expdesign:
    """
    Expdesign class generates stimulus trains for events and convolves them with HRF
    and adds noise to form a realisitic brain signal evoked from an experiment.

    Currently supports only two events A and B and limited TTPs. 
    A future release will extend these limitations for full support

    Attributes:
        create_jitter: Generates jitter distribution as per user input
        
        tcourse: Creates event timecourses based on event trains
    """
    
    def __init__(self,
                 l,
                 u,
                 edur,
                 loadvolume,
                 signal_mag = [2],
                 stim_ratio = None, 
                 null_ratio = None,
                 noise = False,
                 nonlinear = True, 
                 load = None,
                ):
        """
        Initializes the class with user input parameters.
        
        Parameters
        ----------
        l = int
            Lower Inter Stimulus Interval (ISI)
            
        u = int
            Upper Inter Stimulus Interval (ISI)
            
        edur = float
            Event duration
            
        signal_mag = int
            Magnitude of change in signal from baseline
            
        loadvolume = obj
        Noise volume extracted from other MRI file using fmrisim
        
        stim_ratio = float
        Range= 0-1, amplitude of A = stim_ratio * amplitude of B
        
        null_ratio = float
        Range = 0-1, percentage of B events to be set as null
        
        noise = Boolean, True or False
        Yes or No to include fMRI noise in the timecourse or not
        
        nonlinear = Boolean, True or False
        Yes or No to include nonlinear interactions in the event time courses or not
        
        load = string
        'attn': Use attentional mechansisms of transient temporal maintenance like activity during intervals
        'wm': Use working memory mechansisms of transient temporal maintenance like activity during intervals
        None: Not use anything
        
        
        Returns: 
        None.
        """
        self.lower_isi = l
        self.upper_isi = u
        self.burn_in = 5
        self.onsets_A = np.empty((0,1))
        self.onsets_B = np.empty((0,1))
        self.onsets_all = np.empty((0,1))
        self.cue_ratio = stim_ratio # Value between 0-1
        self.null_ratio = null_ratio
        self.temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated?
        self.event_duration = edur  # How long is each event
        self.loadvolume = loadvolume
        self.signal_magnitude = signal_mag
        self.noise = noise
        self.nonlin = nonlinear
        self.load = load


    
    def transient(self,etrain,etrain2,l,u):
        """
        Generates height adjusted stimweight array of event B with transient properties
        expanding the methods of Ruge et al 2009.
        Parameters
        ----------
        def transient : ndarray
        etrain = default weighted stimfunction array for A event
        etrain2 = default weighted stimfunction array for B event
        l = lower bound of isi
        u = upper bound of isi
        self.load = type of transient activity
            'attn'- uses the default attention maintenance map for determining
            the amount of maintenance
            'wm'- uses the default WM maintenance map for determining
            the amount of maintenance
            'wmflat'- special case when neurons fire at an uniform and even rate
            between two events  namely cue and target
        Returns: 
        Modified height adjusted stimweight array of event B with transient properties
        -------
        None.
        """
        tr = int(self.temporal_res)
        sub_imp = 0.8
        for i in np.arange(0,len(etrain)):
            if etrain[i] == 1:
                if self.load == 'attn':
                    
                    if l <= 7 and u <= 9:
                        etrain[ i + 1 : i + 1 + tr * 1 ] = sub_imp                    
                    elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
                          or (l == 8 and u >= 8 and u <= 9)
                          or (l == 9 and u == 9)):
                        etrain[ i + 1 : i + 1 + tr * 4 ] = sub_imp                    
                        
                    elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
                          or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
                          or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
                        etrain[ i + 1 + tr*int(l/2)  : i + 1 + tr*int(l/2) + tr*1 ] = sub_imp                    
        
                        
                    elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
                        or ((l == 10 or l == 11) and u >= 14 and u <= 20)
                        or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
                        or ((l == 14 or l == 15) and u >= 15 and u <= 20)
                        or (l >= 16 and u >= 15 and u <= 20)):
                        etrain[ i + 1 + tr*int(l/2)  : i + 1 + tr*int(l/2) + tr*4 ] = sub_imp
                        
                    return etrain
                elif self.load == 'wm':
                    idxA = np.where(etrain == 1)[0][:]
                    idxB = np.where(etrain2 == 1)[0][:]
                    minl = min(len(idxA),len(idxB))
                    for i in range(minl):
                        if idxA[i] > idxB[i]:
                            idxB[i] = 0  #This is used to treat for any error 
                            # sampling, or when the two events are not alternate
                    
                    if ((l <= 9 and u <= 13) or (l <= 5)):
                        for i in range(minl):
                            if idxB[i]!= 0:
                                etrain2[ idxA[i] + 1 : idxB[i] ] = sub_imp
                                
                    else:
                        for i in range(minl):
                            if idxB[i]!= 0:
                                etrain2[ idxA[i] + 1 + tr*int(l/2) : idxB[i] ] = sub_imp 
                    
                    return etrain2

                
                else:
                    raise ValueError('Invalid transient map arguement')

    def create_jitter(self,distribution,dist_param):
        """
        Creates deterministic jitter distribution probabilites
        for ISI between event A and B based on inputs, incorporates
        distributions discussed in Friston et al 2000.
        Parameters
        ----------
        
        distribution = string
            type of jitter
            'exp'- exponential probability distribution
            'stochastic_rapid'- Stochastic sinusoidal rapid probability distribution
            'stochastic_interm'- Stochastic sinusoidal intermediate probability distribution
            'stochastic_slow'- Stochastic sinusoidal slow probability distribution
            'uniform'- Default. Uniform probability distribution
        dist_param = integer
            Custom parameter for the type of distribution.
        Returns: 
        Weights for probability distribution
        -------
        None.
        """
        
        if distribution not in ['exp','uniform','stochastic_rapid','stochastic_interm','stochastic_slow']:
            raise ValueError('Invalid Distribution')
            
        if self.lower_isi != self.upper_isi and distribution != 'uniform':
            a = np.arange(self.lower_isi,self.upper_isi,0.1)

            if distribution == 'exp':
                ai = np.arange(a.size)        # an array of the index value for weighting

                if dist_param:
                    w = np.exp(ai/dist_param)            # higher weights for larger index values
                else:
                    w = np.exp(ai/30)            # higher weights for larger index values


            elif distribution == 'stochastic_rapid':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if dist_param:
                    w = np.cos(ai/dist_param)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/50))  #50 is chosen for rapid cycles
                w = w + + abs(min(w))


            elif distribution == 'stochastic_interm':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if dist_param:
                    w = np.cos(ai/dist_param)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/20))  #20 is chosen for interm cycles
                w = w + + abs(min(w))

            elif distribution == 'stochastic_slow':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if dist_param:
                    w = np.cos(ai/dist_param)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/6))  #6 is chosen for one slow cycle
                w = w + + abs(min(w))
            
            w /= w.sum()                 # weight must be normalized
            
            self.w = w
            self.a = a
            return w
            
        elif distribution == 'uniform' and self.lower_isi != self.upper_isi:
            a = np.arange(self.lower_isi,self.upper_isi,0.1)
            w = np.ones(a.size)
            w/= w.sum()
            self.w = w
            self.a = a
            
            return w
        
        

    def tcourse(self,hrf_type ,params = None):
        """
        Generates the timecourses based on inputs using the methods of fmrisim (Ellis et al 2020)
        Parameters
        ----------
        
        hrf_type = string
            Type of HRF to convolve the stimulus train with. Currently supports
            double_gamma (default),
            and square
            custom- add custom parameters for double gamma HRF
        params = dict
            Dictionary of parameters as 
            {
            params["response_delay"],
            params["undershoot_delay"],
            params["response_dispersion"],
            params["undershoot_dispersion"],
            params["response_scale"],
            params["undershoot_scale"]
            }
            
        Returns: 
        Timecourse signal convolved with HRF and noise added using fmrisim
        -------
        None.
        """
        pattern_A = self.cue_ratio * np.ones((27,1))
        pattern_B = np.ones((27,1))
        time = self.burn_in
        f = 0 # Variable to switch between two conditions 
        nevents = 0
        
            
        total_time = int(self.loadvolume.dim[3] * self.loadvolume.tr)   # How long is the total event time course
        while time <= (total_time ) :
            if self.lower_isi == self.upper_isi:
                
                if f == 0:
                    self.onsets_A = np.append(self.onsets_A, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + self.lower_isi
                    f = 1
                    nevents = nevents + 1
                else:
                    self.onsets_B = np.append(self.onsets_B, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + self.lower_isi
                    f = 0
                    nevents = nevents + 1
            else:
                        
                if f == 0:
                    self.onsets_A = np.append(self.onsets_A, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + np.random.choice(self.a, size=1, p=self.w)                    
                    
                    f = 1
                    nevents = nevents + 1
                    
                else:
                    self.onsets_B = np.append(self.onsets_B, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + np.random.choice(self.a, size=1, p=self.w)
                   
                    f = 0
                    nevents = nevents + 1

        
        #total_time = time

        self.onsets_A = self.onsets_A.transpose()
        self.onsets_B = self.onsets_B.transpose()
        
        if self.null_ratio != 0 and self.null_ratio != None:
            self.onsets_B = np.sort(np.random.choice(self.onsets_B,
                                                     int(len(self.onsets_B)*self.null_ratio)
                                                     ,replace = False))
            
        stimfunc_A = np.empty((0,1))
        stimfunc_B = np.empty((0,1))


        stimfunc_A = fmrisim.generate_stimfunction(onsets=self.onsets_A,
                                                   event_durations=[self.event_duration],
                                                   total_time=total_time,
                                                   temporal_resolution=self.temporal_res,
                                                   )

        stimfunc_B = fmrisim.generate_stimfunction(onsets=self.onsets_B,
                                                   event_durations=[self.event_duration],
                                                   total_time=total_time,
                                                   temporal_resolution=self.temporal_res,
                                                   )
        
        #Transient activity introduced
        if self.load == 'wmmap':
            stimfunc_B = self.transient(stimfunc_A,stimfunc_B,self.lower_isi,self.upper_isi)
        elif self.load == 'attnmap':
            stimfunc_A = self.transient(stimfunc_A,stimfunc_B,self.lower_isi,self.upper_isi)

            # Multiply each pattern by each voxel time course
        weights_A = np.empty((0,1))
        weights_B = np.empty((0,1))
        weights_B1 = np.empty((0,1))

        weights_A = np.matlib.repmat(stimfunc_A, 1, self.loadvolume.voxels).transpose() * pattern_A
        weights_B = np.matlib.repmat(stimfunc_B, 1, self.loadvolume.voxels).transpose() * pattern_B
        weights_B1 = np.matlib.repmat(stimfunc_B, 1, self.loadvolume.voxels).transpose() * pattern_B


        
        # Sum these time courses together
        stimfunc_weighted = np.empty((0,1))

        stimfunc_weighted = weights_B1 + weights_A
        stimfunc_weighted = stimfunc_weighted.transpose()
        self.temp = stimfunc_weighted
        

        signal_func = fmrisim.convolve_hrf(stimfunction=stimfunc_weighted,
                                           tr_duration=self.loadvolume.tr,hrf_type = hrf_type,params = params,
                                           temporal_resolution=self.temporal_res,
                                           scale_function=0,nonlin = self.nonlin)

        # Specify the parameters for signal
        signal_method = 'PSC'

        # Where in the brain are there stimulus evoked voxels
        # The np.where is traversing through the self.signal_volume across every coordinate and looking where the signal is to be 
        # evoked as set before
        signal_idxs = np.where(self.loadvolume.signal_volume == 1)

        # Pull out the voxels corresponding to the noise volume
        noise_func = self.loadvolume.noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :]

        # Compute the signal appropriate scaled
        signal_func_scaled = fmrisim.compute_signal_change(signal_func,
                                                           noise_func.T,
                                                           self.loadvolume.noise_dict,
                                                           magnitude=self.signal_magnitude,
                                                           method=signal_method,)
        
        signal = fmrisim.apply_signal(signal_func_scaled,self.loadvolume.signal_volume,)
        self.temp = signal
        if self.noise:
            
            self.brain = signal + self.loadvolume.noise
        else:
            self.brain = signal
        
        return self.brain

    
class expanalyse:
    
    """
    Expanalyse class provides statistical tools to analyze the efficiency of the created time course.

    Currently supports only two events A and B and limited TTPs. 
    A future release will extend these limitations for full support

    Attributes:
        calc_Fd: Calculates detection power of an event
        
        calc_Fe: Calculates estimatin efficiency of an event
    """
     
    def __init__(self,data,contrast,expdesign):
        
        """
        Initializes the class with user input parameters.
        
        Parameters
        ----------
        data = Output from expdesign.tcourse()
            Modelled timecourse of events for analysis
            
        contrast = list
        [1 0]: Optimality check for event A
        [0 1]: Optimality check for event B
        [0.5 0.5]: Check for both events
        [1 1]: Check for both with respect to baseline
        
        expdesign: Object of expdesign
        To pass on design parameters
        This is deprecated. In a future release, this will not be required
        
        
        Returns: 
        None.
        """
         
        dist_paramdesign = expdesign

        HRF = _dghrf()
        self. boxcar_A = np.zeros(np.size(data,3))
        self. boxcar_B = np.zeros(np.size(data,3))
        
        self.boxcar_A[(dist_paramdesign.onsets_A/dist_paramdesign.loadvolume.tr).astype('int')] = 1
        self.conv_boxcar_A = np.convolve(self.boxcar_A,HRF,'same')
       
        self.boxcar_B[(dist_paramdesign.onsets_B/dist_paramdesign.loadvolume.tr).astype('int')] = 1
        self.conv_boxcar_B = np.convolve(self.boxcar_B,HRF,'same')
        
        lb = (dist_paramdesign.loadvolume.coordinates - ((dist_paramdesign.loadvolume.feature_size - 1) / 2)).astype('int')[0]
        ub = (dist_paramdesign.loadvolume.coordinates + ((dist_paramdesign.loadvolume.feature_size - 1) / 2) + 1).astype('int')[0]
        
        roi_brain = data[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2], :]
        roi_brain = roi_brain.reshape((dist_paramdesign.loadvolume.voxels,data.shape[3]))
        self.roi = roi_brain[13,:].T
        
        self.X = np.zeros((self.conv_boxcar_A.shape[0],2))
        self.design = np.empty((self.boxcar_A.shape[0],2))
        self.X[:,0] = self.conv_boxcar_A
        self.X[:,1] = self.conv_boxcar_B
        self.design[:,0] = self.boxcar_A
        self.design[:,1] = self.boxcar_B
        
        self.C = contrast
        
        
    def calc_Fd(self):
        
        """
        Calculates detection power of an event in a timecourse
        
        Parameters
        ----------
        None
        
        
        Returns: 
        Detection power of an event in a timecourse 
        """

        temp = sm.OLS(self.roi,sm.add_constant(self.X)).fit()
        res_resid = sm.OLS(temp.resid[1:], temp.resid[:-1]).fit()
        rho = res_resid.params
        order = toeplitz(np.arange(len(self.roi)))
        sigma = rho**order
        gls_model = sm.GLS(self.roi, self.X, sigma=sigma)
        gls_results = gls_model.fit()
        med = gls_model.cholsigmainv * sigma * gls_model.cholsigmainv.T
        m = np.dot(np.dot(gls_model.pinv_wexog , med) , gls_model.pinv_wexog.T)
        o = self.C*m*self.C.T
        out = 1/np.matrix.trace(o)
        return out
    
    def calc_FIR(self,ncond,hrflen = 30):
        """
        Calculates Finite Impulse responses for an HRF
        
        Parameters
        ----------
        ncond = int
        Number of unique events (2 for now)
        
        hrflen = int
        Length of HRF modelled
        
        
        Returns: 
        Finite Impulse responses for an HRF.
        
        """
        nCond = ncond
        hrflen = hrflen
        self.X_FIR = np.zeros((len(self.roi),hrflen*nCond))
        
        for iC in range(0,nCond):
            temp = np.nonzero(self.design[:,iC])
            onsets = []
            for item in temp:
                onsets.extend(item)
    
            col = iC + 1
            idxCols = np.arange(((col-1)*hrflen+1),(col*hrflen))
            for jO in range(0,len(onsets)-1):
                idxRows = np.arange(onsets[jO],(onsets[jO]+hrflen-1))
                for kR in range(0,len(idxRows)):
                    if idxRows[kR] < len(self.roi):
                        self.X_FIR[int(idxRows[kR]),int(idxCols[kR])] = 1
                    else:
                        break
        return self.X_FIR
                    
                    
    def calc_Fe(self,ncond,hrflen = 30):
        
        """
        Calculates estimation efficiency of an event in a timecourse
        
        Parameters
        ----------
        None
        
        
        Returns: 
        Estimation efficiency of an event in a timecourse.
        
        """
        
        hrflen = hrflen
        X_FIR = self.calc_FIR(ncond)
        temp = sm.OLS(self.roi,sm.add_constant(X_FIR)).fit()
        res_resid = sm.OLS(temp.resid[1:], temp.resid[:-1]).fit()
        rho = res_resid.params
        order = toeplitz(np.arange(len(self.roi)))
        sigma = rho**order
        gls_model = sm.GLS(self.roi, X_FIR, sigma=sigma)
        gls_results = gls_model.fit()
        med = gls_model.cholsigmainv * sigma * gls_model.cholsigmainv.T
        m = np.dot(np.dot(gls_model.pinv_wexog , med) , gls_model.pinv_wexog.T)
        C = np.array(np.kron(self.C, np.eye(hrflen)))
        o = np.dot(np.dot(C,m),C.T)
        out = 1/np.matrix.trace(o)
        return out
    
    

        
         
         

