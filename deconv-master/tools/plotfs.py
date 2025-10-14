#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:55:51 2020

This module provides a set of functions to help vizualize and plot
different outputs related to timecourses and their optimality.


@author: Soukhin Das (skndas@ucdavis.edu), Center for Mind and Brain, Davis, California
@author: Weigang Yi, Center for Mind and Brain, Davis, California
University of California, Davis
"""

__all__ =[
    'plot_result',
    'plot_weighted_result',
    'plot_jitter_dist',
    'plot_avg_response',
    'graph_timecourses'
    
]


import matplotlib.pyplot as plt
import numpy as np
import copy
import itertools


def plot_result(data1,data2,normalize = False,title1 = "Detection Power",
             title2 = "Estimation Efficiency"):
    """
        Plots optimality figures across combinations of lisi and uisi
        
        Parameters
        ----------
        data1 = nxn float array
        Detection power
        
        data2 = nxn float array
        Estimation efficiency
        
        normalize = True or False (Optional)
        Max normalizes the output between 0-1
        
        title1 = string
        Optional title for detection power plot
        
        title2 = string
        Optional title for estimation efficiency plot
        
        ----------
        Returns: 
        fig1,fig2 figure axes
        
    """
    
    max1 = np.amax(data1)
    max2 = np.amax(data2)
    if normalize:
        data1 /=np.amax(data1)
        data2 /=np.amax(data2)
        max1 = 1
        max2 = 1
        
        
    fig1= plt.figure(figsize = (5,5))
    ax = fig1.add_subplot(111)
    im = ax.imshow(data1,vmin = 0, vmax = max1,cmap = 'viridis')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title1,size = 15)
    fig1.colorbar(im)
    labels = np.arange(1,len(data1)+1,2).astype(str)
    plt.xticks(np.arange(0,len(data1),2), labels)
    plt.yticks(np.arange(0,len(data1),2), labels)

    fig2= plt.figure(figsize = (5,5))

    ax = fig2.add_subplot(111)
    im2 = ax.imshow(data2, vmin = 0, vmax = max2, cmap = 'viridis')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title2,size = 15)
    fig2.colorbar(im2)
    plt.xticks(np.arange(0,len(data2),2), labels)
    plt.yticks(np.arange(0,len(data2),2), labels)
    
    return fig1,fig2


def plot_weighted_result(data1, data2, weights = [ 0.5 , 0.5 ], title = "Weighted Optimality"):
    
    """
        Plots weighted optimality figure across combinations of lisi and uisi using user input weights.
        
        Parameters
        ----------
        data1 = nxn float array
        Detection power
        
        data2 = nxn float array
        Estimation efficiency
        
        weights = 1 x 2 int list
        [w1 , w2], w1 weighing data1, w2 weighing data2
        Must: w1 + w2 = 1
        
        title = string
        Optional title for plot
        
        
        ----------
        Returns: 
        fig figure axes
        
    """
    
    if np.sum(weights) != 1:
        raise ValueError('The sum of weights is not equal to 1')

    data1 = data1 / np.amax(data1)
    data2 = data2 / np.amax(data2)

    data = weights[0] * data1 + weights[1] * data2

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, vmin = 0, vmax = 1, cmap = 'viridis')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title,size = 15)
    fig.colorbar(im)
    labels = np.arange(1,len(data)+1,2).astype(str)
    plt.xticks(np.arange(0,len(data),2), labels)
    plt.yticks(np.arange(0,len(data),2), labels)

    return fig



def plot_jitter_dist(w):
    
    """
        Plots the jitter probability distribution of the ISI
        
        Parameters
        ----------
        w = list
        Probability distribution of the ISI jitter
        
        
        ----------
        Returns: 
        None
    """
    
    fig1 = plt.figure(figsize = (5,5))
    ax = plt.plot(w)
    plt.yticks([0,max(w)],['0','1'])
    plt.ylim([0,max(w)*1.2])
    plt.xticks([0,len(w)],['Lower ISI','Upper ISI'])
    plt.xlabel('ISI')
    plt.ylabel('Probability')
    plt.title('Occurence probability')

def plot_avg_response(onsets, brain, before = 2, after = 10):
    """
        Plots an epoch of event time course
        
        Parameters
        ----------
        onset = list
        Onsets of events
        
        brain = list
        timecourse of the signal
        
        before = int
        How many TRs before should the epoch start.
        
        after = int
        How many TRs after should the epoch end.
        
        
        ----------
        Returns: 
        None
    """
    
    out = avg_response(onsets,brain, before,after)
    out = out / np.max(out)
    plt.figure(figsize = (5,5))
    plt.plot(out)
    ooo = np.arange(-before,after+1,2)
    oo = np.arange(-before,after+1,1)
    plt.xticks(np.arange(0,len(out),2),labels = ooo.astype(str),fontsize = 12)
    plt.yticks([0,0.5,1],'')
    plt.ylim(-0.15,1.2)
    plt.xlim(0,len(out))
    plt.axhline(0, color='red',lw = 1)
    plt.axvline(np.where(oo == 0) , color='red',lw = 1)
    plt.title('Average evoked response',fontsize = 15)
    plt.xlabel('TR',fontsize = 15)
    plt.ylabel('Amplitude',fontsize = 15)


def graph_timecourses(result, time_courses, stimuli = [ 'A', 'B' ], xlim = 50, ylim = 1.3):
    
    """
    Plots an epoch of event time course

    Parameters
    ----------
    result = list of lists
    All timecourses obtained from run_experiment

    time_courses = list of list
    Pair of lisi and uisi to extract the signal from

    stimuli = list of string
    Name of the two events

    xlim = int
    Length of the timecourse to plot on x axis

    ylim = int
    Length of amplitude to plot on y axis

    ----------
    Returns: 
    None
    """
        
        
    e_t_array = []
    
    for time_course in time_courses:
        e_t = copy.deepcopy(result[str(time_course[0])][str(time_course[1])])

        t = remove_transient(e_t["t"])
        if 'A' not in stimuli:
            t = remove_irrelevant(t, 'A')
        elif 'B' not in stimuli:
            t = remove_irrelevant(t, 'B')

        e_t_array = [ *e_t_array, {
            "e": e_t["e"] / np.max(e_t["e"]),
            "t": t, 
        } ]

    fig = plt.figure(figsize = (20,10))

    num_figures = len(time_courses)


    mainax = fig.add_subplot(111)
    mainax.spines['top'].set_color('none')
    mainax.spines['bottom'].set_color('none')
    mainax.spines['left'].set_color('none')
    mainax.spines['right'].set_color('none')
    mainax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    cmap = plt.cm.get_cmap('hsv', num_figures + 10)
    for i in range(0, num_figures):
        ax = fig.add_subplot(int(num_figures), 1, i + 1)
        e_t = e_t_array[i]
        plt.plot(e_t["e"], color = cmap(i))
        plt.xlim(0, xlim)
        plt.ylim(-0.1, ylim + 0.2)
        (markers, stemlines, baseline) = plt.stem(e_t["t"])
        plt.setp(stemlines, linestyle="-", color="black", linewidth=1 )
        plt.setp(markers, color="black")

        plt.xticks(np.linspace(0,xlim,11),fontsize = 12)
        plt.yticks([0,0.5,1],labels = [],fontsize = 12)

        #ax.set_title('Profile 5',fontsize = 15)
        f = 1
        for i in range(0,xlim):
            if e_t["t"][i] == 1:
                marker = 'A'
                if f == 1:
                    marker = 'A'
                    if 'A' not in stimuli:
                        marker = 'B'
                elif f == 0:
                    marker = 'B'
                    if 'B' not in stimuli:
                        marker = 'A'
                ax.text(i - 0.1, 1.1, marker, fontsize = 12)
                f = f + 1
                f = f % 2

    mainax.set_xlabel('TR',fontsize = 18)
    mainax.set_ylabel('Amplitude', fontsize = 18)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    #fig.savefig("timecourses",dpi = 600,bbox_inches = 'tight',pad_inches = 0.1)

    return fig

def plot_null_ratios(const_lisi,
                     lim_uisi,
                     null_ratios,
                     det,
                     est):


    num_lines = len(det.T)
    cmap = plt.cm.get_cmap('tab10',num_lines + 5)
    legend_names = null_ratios.astype(str)
    xlabels = np.arange(const_lisi,lim_uisi + const_lisi + 1,1).astype(str)
    xlabels[0] = xlabels[0] + 's const.'

    fig = plt.figure(figsize = (10,13))
    
    ax = fig.add_subplot(211)
    marker = itertools.cycle(('s', 'v', '.', 'o', '*')) 
    plt.title('Detection power as null ratios',fontsize = 20)
    plt.ylabel('Detetion power',fontsize = 20)
    plt.xticks(np.arange(0,lim_uisi + 1,1),xlabels,fontsize = 10)
    for i in range(num_lines):
        data = det[:,i]
        plt.plot(data.T, marker = next(marker),color = cmap(i),label=legend_names[i])
        plt.legend(title = 'Null ratios')
        
        
    ax = fig.add_subplot(212)
    marker = itertools.cycle(('s', 'v', '.', 'o', '*')) 
    plt.title('Estimation efficiency as null ratios',fontsize = 20)
    plt.ylabel('Estimation efficiency ',fontsize = 20)
    plt.xlabel('Upper Bound of ISI',fontsize = 20)
    plt.xticks(np.arange(0,lim_uisi + 1,1),xlabels,fontsize = 10)
    for i in range(num_lines):
        data = est[:,i]
        plt.plot(data.T, marker = next(marker),color = cmap(i),label=legend_names[i])
        plt.legend(title = 'Null ratios')
        
        
        

def avg_response(onsets, brain, before, after):
    
    """
        Calculates an epoch of event time course
        
        Parameters
        ----------
        onsets = list
        Onsets of events
        
        brain = list of list
        Pair of lisi and uisi to extract the signal from
        
        before = int
        How many TRs before should the epoch start.
        
        after = int
        How many TRs after should the epoch end.
        
        ----------
        Returns: 
        None
    """
    onset_indices = []
    for i in range(0, len(onsets)):
        if onsets[i] == 1:
            onset_indices = [ *onset_indices, i ]
    res = after + before + 1
    out = np.zeros((len(onset_indices), res))
    for i in range(0,len(onset_indices)):
        start = onset_indices[i]
        series = np.linspace(start - before, start + after, res).astype('int')
        if series[-1] >= brain.shape[0]:
            break
        else:
            out[i,:] = brain[series]
        
    out = np.mean(out, 0)
    return out

def remove_transient(x):
    """
        Removes transient profiles for clear plotting
        
        Parameters
        ----------
        x = list of lists
        All timecourses obtained from run_experiment
        
        ----------
        Returns: 
        Clean timecourses
    """
    for i in range(0,len(x)):
        if x[i] == 0.66:
            x[i] = 0
    return x

def remove_irrelevant(x, excluded_stimuli = 'A' ):
    order = 0;
    excluded_order = 0 if excluded_stimuli == 'A' else 1
    for i in range(0, len(x)):
        if x[i] == 1:
            if order == excluded_order:
                x[i] = 0
            order = order + 1
            order = order % 2
    return x

