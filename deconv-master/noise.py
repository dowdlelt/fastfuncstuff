#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:22:01 2020

This code provides a set of functions to parse a separate MRI file to extract noise
from it using the routines provided by fMRIsim (Ellis et al 2020).

@author: Soukhin Das (skndas@ucdavis.edu), Center for Mind and Brain, Davis, California
@author: Weigang Yi, Center for Mind and Brain, Davis, California
University of California, Davis

"""
import fmrisim_modified as fmrisim
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import json

__all__ = [
    'loadvolume'
]

class loadvolume:
    
    """
    Loadvolume class handles the MRI file and provides functions to 
    systematically extract noise from it using fMRIsim (Ellis et al 2020).

    Attributes:
        loaddata: Handles the path of the MRI file and extracts information.
        
        loadmask: Creates a mask for the ROI
        
        generate_noise: Extracts noise parameters from the MRI file
        
        generate_region: Creates a ROI to work with
        
        
    """
    def __init__(self,path):
        
        """
        Constructor
        
        Parameters
        ----------
        path: string
        Path to the MRI file from which noise is extracted.
        
        __________
        Returns: 
        None
        """
        
        self.path = path
    
    def loaddata(self):
        
        """
        Loads the MRI file and generates information about TR, length and dimension of the scan.
        
        Parameters
        ----------
        None
        __________
        Returns: 
        None
        """
        
        nii = nibabel.load(self.path)  #load the pathj
        self.volume = nii.get_data()
        self.dim = self.volume.shape  # What is the size of the volume
        self.dimsize = nii.header.get_zooms()  # Get voxel dimensions from the nifti header
        self.tr = self.dimsize[3]
        if self.tr > 100:  # If high then these values are likely in ms
            self.tr /= 1000
    
    def loadmask(self):
        
        """
        Generates mask around the ROI
        
        Parameters
        ----------
        None
        __________
        Returns: 
        None
        """
        
        self.mask, self.template = fmrisim.mask_brain(volume=self.volume, 
                                    mask_self=True,)
    
        
    def generate_noise(self):
        
        """
        Extracts and calculates noise parameters from the MRI file using
        fMRIsim (Ellis et al 2020)
        
        Parameters
        ----------
        None
        __________
        Returns: 
        None
        """
        
        noise_dict = {'voxel_size': [self.dimsize[0], self.dimsize[1], 
                                     self.dimsize[2]], 'matched': 1}
        self.noise_dict = fmrisim.calc_noise(volume=self.volume,
                                mask=self.mask,
                                template=self.template,
                                noise_dict=noise_dict,
                                )

        # Calculate the noise given the parameters
        self.noise = fmrisim.generate_noise(dimensions=self.dim[0:3],
                               tr_duration=int(self.tr),
                               stimfunction_tr=[0] * self.dim[3], 
                               mask=self.mask,
                               template=self.template,
                               noise_dict=self.noise_dict,
                               )
        
    def generate_region(self):
        
        """
        Creates the ROI that will be used later to store the timecourse and
        add noise.
        
        Parameters
        ----------
        None
        __________
        Returns: 
        None
        """
        
        # Create the region of activity where signal will appear
        self.coordinates = np.array([[21,21,21]])  # Where in the brain is the signal
        self.feature_size = 3  # How big, in voxels, is the size of the ROI
        self.signal_volume = fmrisim.generate_signal(dimensions=self.dim[0:3],
                                                feature_type=['cube'],
                                                feature_coordinates=self.coordinates,
                                                feature_size=[self.feature_size],
                                                signal_magnitude=[1],
                                                )
        plt.figure()
        plt.imshow(self.signal_volume[:,:,21] , cmap=plt.cm.gray)
        plt.imshow(self.mask[:, :, 21], cmap=plt.cm.gray, alpha=.5)
        plt.axis('off')
        self.voxels = self.feature_size ** 3
        

    