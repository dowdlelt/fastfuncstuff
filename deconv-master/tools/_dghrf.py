#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:57:16 2020
Forked from fmrisim (Ellis et. al. 2020)
@author: sdas
"""
import math
from scipy.stats import gamma


def _dghrf(response_delay=6,
                      undershoot_delay=12,
                      response_dispersion=0.9,
                      undershoot_dispersion=0.9,
                      response_scale=1,
                      undershoot_scale=0.035,
                      temporal_resolution=1.0,
                      ):
    """Create the double gamma HRF with the timecourse evoked activity.
    Default values are based on Glover, 1999 and Walvaert, Durnez,
    Moerkerke, Verdoolaege and Rosseel, 2011
    """

    hrf_length = 30  # How long is the HRF being created

    # How many seconds of the HRF will you model?
    hrf = [0] * int(hrf_length * temporal_resolution)

    # When is the peak of the two aspects of the HRF
    response_peak = response_delay * response_dispersion
    undershoot_peak = undershoot_delay * undershoot_dispersion

    for hrf_counter in list(range(len(hrf) - 1)):

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = math.pow((hrf_counter / temporal_resolution) /
                            response_peak, response_delay)
        resp_exp = math.exp(-((hrf_counter / temporal_resolution) -
                              response_peak) /
                            response_dispersion)

        response_model = response_scale * resp_pow * resp_exp

        undershoot_pow = math.pow((hrf_counter / temporal_resolution) /
                                  undershoot_peak,
                                  undershoot_delay)
        undershoot_exp = math.exp(-((hrf_counter / temporal_resolution) -
                                    undershoot_peak /
                                    undershoot_dispersion))

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For this time point find the value of the HRF
        hrf[hrf_counter] = response_model - undershoot_model

    return hrf

def hrf2(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values     
    # Scale max to 0.6
    return values