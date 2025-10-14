import design
import numpy as np
import noise
import time
from tools import plotfs


"""
Created on Nov 15 14:16:41 2022

This module provides a set of functions to conduct investigations 
by manipulating various design parameters as per user input.



@author: Soukhin Das (skndas@ucdavis.edu), Center for Mind and Brain, Davis, California
@author: Weigang Yi, Center for Mind and Brain, Davis, California
University of California, Davis
"""


__all__ = [
    "load_noise",
    "run_experiment"
]

def run_experiment(max_lisi, max_uisi, noise_volume, parameters):
    
    """
        Generates the timecourses and calculates their measures of optimality 
        using the design parameters provided.
        
        Parameters
        ----------
        max_lisi: int
            Maximum bound of the lower ISI to test.
            
        max_uisi: int
            Maximum bound of the upper ISI to test.
            
        noise_volume: Object of noise
            Noise object obtained from load_noise()
            
        parameters: dict
            Dictionary of parameters
            
            {
                "event_duration" : float,
                "signal_mag": int,
                "distribution" : "uniform"/"stochastic_rapid"/"stochastic_interm"/"stochastic_slow"/"exp"
                "dist_param" : int,
                "stim_ratio" : float,0-1 (Optional = 1)
                "noise" :  True or False, (Optional = False)
                "nonlinear" : True or False, (Optional = False)
                "null_ratio": float,0-1 (Optional = 1)
                "transient_load": "attn"/"wm" (Optional = None)
             }
             
        __________
        
        Returns: 
        p1 = Detection power of all combinations of parameters
        p2 = Estimation efficiency of all combinations of parameters
        result = Timecourses of all combinations of parameters
        
    """
    
    
    # Initialize empty list to store efficiency measures
    p1 = np.zeros((max_lisi, max_uisi))
    p2 = np.zeros((max_lisi, max_uisi))
    result = {}

    start = time.time()
    
    k = 0
    store = 0

    #Start a loop to iterate through all combinatins of lisi and uisi
    for lisi in np.arange(1, max_lisi + 1, 1):
        result[str(lisi)] = {}
        l = 0
        for uisi in np.arange(1, max_uisi + 1, 1):
            if lisi > uisi:
                l += 1
                continue

            #Create a expdesign object with all the set parameters
            d = design.expdesign(
                         lisi,
                         uisi,
                         parameters["event_duration"], 
                         noise_volume,  
                         parameters["signal_mag"], 
                         stim_ratio = parameters["stim_ratio"],
                         null_ratio = parameters["null_ratio"], 
                         noise = parameters["noise"], 
                         nonlinear = parameters["nonlinear"], 
                         load = parameters["transient_load"]
                        )
            # Create a jitter distribution that will be used to determine the nature of ISI
            w = d.create_jitter(parameters["distribution"], 
                                parameters["dist_param"])
            
            if uisi == max_uisi - 1 and store == 0:
                w_store = w
                store = 1
                
            
            #Generate the timecourse
            data = d.tcourse(parameters["hrf_type"],parameters["hrf_params"])
            
            #Calculate the optimality measures of the generated timecourse
            e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
            
            #Store the optimality measures 
            p1[k,l] = e.calc_Fd()
            p2[k,l] = e.calc_Fe(ncond =2)

            #Store the timecourses generated
            result[str(lisi)][str(uisi)] = {
                "e": e.roi,
                "t": e.design[:,0] + e.design[:,1],
                "t1": e.design[:,0],
                "t2": e.design[:,1]
            }

            l += 1
        k += 1
        
    print(f'Time elapsed: {time.time() - start}')
    plotfs.plot_jitter_dist(w_store)

    return p1, p2, result
    
def run_null_experiment(constant_lisi, max_uisi,null_ratios, noise_volume, parameters):
    
    """
        Generates the timecourses and calculates their measures of optimality 
        using the design parameters provided for null events.
        
        Parameters
        ----------
        constant_lisi: int
            Constant value of lower ISI to test.
            
        max_uisi: int
            Maximum bound of the upper ISI to test.
        null_ratios: list of int
            List of ratios that will set the ratio of B events to null
            
        noise_volume: Object of noise
            Noise object obtained from load_noise()
            
        parameters: dict
            Dictionary of parameters
            
            {
                "event_duration" : float,
                "signal_mag": int,
                "distribution" : "uniform"/"stochastic_rapid"/"stochastic_interm"/"stochastic_slow"/"exp"
                "dist_param" : int,
                "stim_ratio" : float,0-1 (Optional = 1)
                "noise" :  True or False, (Optional = False)
                "nonlinear" : True or False, (Optional = False)
                "null_ratio": float,0-1 (Optional = 1)
                "transient_load": "attn"/"wm" (Optional = None)
             }
             
        __________
        
        Returns: 
        p1 = Detection power of all combinations of parameters
        p2 = Estimation efficiency of all combinations of parameters
        result = Timecourses of all combinations of parameters
        
    """
    
    
    # Initialize empty list to store efficiency measures
    p1 = np.zeros((len(null_ratios), max_uisi-constant_lisi + 1))
    p2 = np.zeros((len(null_ratios), max_uisi-constant_lisi + 1))
    result = {}

    start = time.time()

    #Start a loop to iterate through all combinatins of lisi and uisi
    lisi = constant_lisi
    k = 0
    for ratio in null_ratios:
        result[str(lisi)] = {}
        l = 0
        for uisi in np.arange(lisi, max_uisi + 1, 1):
            if lisi > uisi:
                l += 1
                continue

            #Create a expdesign object with all the set parameters
            d = design.expdesign(
                         lisi,
                         uisi,
                         parameters["event_duration"], 
                         noise_volume,  
                         parameters["signal_mag"], 
                         stim_ratio = parameters["stim_ratio"],
                         null_ratio = ratio, 
                         noise = parameters["noise"], 
                         nonlinear = parameters["nonlinear"], 
                         load = parameters["transient_load"]
                        )
            # Create a jitter distribution that will be used to determine the nature of ISI
            w = d.create_jitter(parameters["distribution"], 
                                parameters["dist_param"])


            #Generate the timecourse
            data = d.tcourse(parameters["hrf_type"],parameters["hrf_params"])

            #Calculate the optimality measures of the generated timecourse
            e = design.expanalyse(data, np.array([1, 0]), expdesign = d)

            #Store the optimality measures 
            p1[k,l] = e.calc_Fd()
            p2[k,l] = e.calc_Fe(ncond =2)

            #Store the generated timecourses 
            result[str(lisi)][str(uisi)] = {
                "e": e.roi,
                "t": e.design[:,0] + e.design[:,1],
                "t1": e.design[:,0],
                "t2": e.design[:,1]
            }

            l += 1
        k += 1

    print(f'Time elapsed: {time.time() - start}')
    #plotfs.plot_jitter_dist(w_store)

    return p1, p2, result
    

    
def load_noise(noise_file_path):
    
    """
        Loads the MRI file path from which noise will be extracted.
        
        Parameters
        ----------
        noise_file_path: string
        Path to the MRI file, .nii supported
        
        __________
        Returns: 
        Object containing information about the extracted noise.
    """
    
    #Parse the path for information about the scan
    lv = noise.loadvolume(noise_file_path)
    
    #Load the data
    lv.loaddata()
    
    #Load the mask around the brain
    lv.loadmask()
    
    #Calcuate the noise from the file
    lv.generate_noise()
    
    #Calculate the ROI where the timecourse will be stored
    lv.generate_region()
    
    return lv