import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy
import random
import pandas as pd
import random
import re
import os
from scipy.stats import norm
import statistics

import psrqpy
from copy import deepcopy
import seaborn as sns
from joblib import Parallel, delayed, Memory 
from functools import lru_cache
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from parallelbar import progress_imap, progress_map, progress_imapu
from parallelbar.tools import cpu_bench, fibonacci
import scipy.stats as stats

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
from matplotlib.patches import Patch  # For proxy legend entry
from astropy.coordinates import CartesianRepresentation, SkyCoord
from matplotlib.lines import Line2D

class TimeSeries:
    def __init__(self, on_times=[], off_times=[], period=0.0, extended = False):
        """
        Initialize a TimeSeries instance.

        :param on_times: A list of floats representing on times. Default is an empty list.
        :param off_times: A list of floats representing off times. Default is an empty list.
        :param period: A float representing the period. Default is 0.0.
        """
        self.on_times = on_times 
        self.off_times = off_times
        self.period = period
        self.extended = extended

    def __repr__(self):
        """
        Return a string representation of the TimeSeries instance.
        """
        return (
            f"TimeSeries(on_times={self.on_times}, "
            f"off_times={self.off_times}, "
            f"period={self.period})"
        )
    
    def shift(self, dt):
        """
        Shift the on and off times by a given time step.

        :param dt: A float representing the time step.
        :return: A new TimeSeries instance with shifted on and off times.
        """
        return TimeSeries(
            on_times=[(t + dt) % self.period for t in self.on_times],
            off_times=[(t + dt) % self.period for t in self.off_times],
            period=self.period
        )

    def extend_N_times(self, N):
        """
        Copy the on and off times N times (keeping periodicity).

        :param N: An integer representing the number of times to extend.
        :return: A new TimeSeries instance with extended on and off times.
        """
        if self.extended == True:
            raise ValueError("TimeSeries instance is already extended. Either this is an error, or you need to code a new extend method.")
        
        return TimeSeries(
            on_times=[t + i * self.period for i in range(N) for t in self.on_times],
            off_times=[t + i * self.period for i in range(N) for t in self.off_times],
            period=self.period * N,
            extended = True
        )

# The primary function, gives a cost
def overlap_cost(X,Y):
    """
    Calculate the maximum overlap cost between two TimeSeries instances over all possible phase offsets.

    :param X: A TimeSeries instance.
    :param Y: A TimeSeries instance.
    :return: A float representing the overlap cost.
    """
    
    max_overlap = 0. # initialize the maximum overlap cost
    DT = 0. # initialize the time offset integration

    Xmin = np.min([X.period,Y.period])==X.period # Tracks if X has the smaller period

    min_period = np.min([X.period, Y.period])
    prev_X, prev_Y = None, None
    prev_overlap = None
    while DT <= min_period:
        
        dt = t_step(X,Y)
        #print("t+step is ",dt)
        DT += dt
        #print('DT is ',DT)
        if dt ==np.inf:
            w=2 ### place holder value to ensure that code continues to run even when dt = inf and take up little memory
        elif Xmin:
            #print('Xmin is true and shifting next')
            X, Y = X.shift(dt), Y
            #print(X,Y)
            #print('shift successful', X)
        else:
            #print('Xmin is false and shifting next')
            X, Y = X, Y.shift(dt)
            #print(X,Y)
            #print('shift successful',X)
    
        if prev_X == X and prev_Y == Y:
            overlap = prev_overlap
        else:
            overlap = eval_overlap(X,Y)
            #print('this is overlap',overlap)
            #print('overlap successful',overlap)
            prev_X, prev_Y = X, Y
            prev_overlap = overlap
        max_overlap = np.max([overlap, max_overlap])

    return max_overlap

def t_step(X,Y):
    """
    Calculate the time step between two TimeSeries instances...

    :param X: A TimeSeries instance.
    :param Y: A TimeSeries instance.
    :return: A float representing the time step.
    """

    Xmin = np.min([X.period,Y.period]) == X.period # Tracks if X has the smaller period
    dt = np.inf # initialize the time step
    
    if Xmin:
        for x_on in X.on_times:
            for y_on in Y.on_times:
                if x_on < y_on:
                    dt = np.min([y_on - x_on, dt])
                else:
                    dt = np.min([y_on + Y.period - x_on, dt])
        for x_off in X.off_times:
            for y_off in Y.off_times:
                if x_off < y_off:
                    dt = np.min([y_off - x_off, dt])
                else:
                    dt = np.min([y_off + Y.period - x_off, dt])
    else:
        for x_on in X.on_times:
            for y_on in Y.on_times:
                if y_on < x_on:
                    dt = np.min([x_on - y_on, dt])
                else:
                    dt = np.min([x_on + X.period - y_on, dt])
        for x_off in X.off_times:
            for y_off in Y.off_times:
                if y_off < x_off:
                    dt = np.min([x_off - y_off, dt])
                else:
                    dt = np.min([x_off + X.period - y_off, dt])
    return dt

def eval_overlap(X,Y):
    """
    Calculate the maximum overlap length between two TimeSeries instances.

    :param X: A TimeSeries instance.
    :param Y: A TimeSeries instance.
    :return: A float representing the maximum overlap length.
    """
    overlap_count = 0
    change_events = 0
    sub_strings=[]

    small_always_on= False
    small_always_off = False
    small_both_off=False

    big_always_on=False
    big_always_off=False
    big_both_off=False

    Xmin = np.min([X.period,Y.period]) == X.period # Tracks if X has the smaller period
    
    if Xmin:
        #zero=np.arange(0,1,1)
        if len(Y.off_times)<=0:

            small_times= np.arange(0,((3*Y.period)),X.period)
            big_times= np.arange(0,((3*Y.period)),Y.period)
        else:
            small_times= np.arange(0,((3*Y.period)+Y.off_times[0]),X.period)
            big_times= np.arange(0,((3*Y.period)+Y.off_times[0]),Y.period)

        if len(X.on_times)<=0 and len(X.off_times)>0:
            small_always_off = True
        if len(X.on_times)>0 and len(X.off_times)<=0:
            small_always_on = True
        if len(X.on_times)<=0 and len(X.off_times)<=0:
            small_both_off = True

        if len(Y.on_times)<=0 and len(Y.off_times)>0:
            big_always_off = True
        if len(Y.on_times)>0 and len(Y.off_times)<=0:
            big_always_on = True
        if len(Y.on_times)<=0 and len(Y.off_times)<=0:
            big_both_off = True

        if small_always_off == True and small_both_off == False:
            small_off_times=X.off_times[0]+small_times  
            small_on_times=np.array(['n'])

        if small_always_on == True:
            small_off_times=np.array(['n'])
            small_on_times = X.on_times[0]+small_times

        if small_both_off == True:
            small_off_times=np.array(['n'])
            small_on_times=np.array(['n'])
        
        if big_always_off == True and big_both_off == False:
            big_off_times=Y.off_times[0]+big_times
            big_on_times=np.array('n')

        if big_always_on == True:
            big_off_times=np.array(['n'])
            big_on_times=Y.on_times[0]+big_times

        if big_both_off == True:
            big_off_times=np.array(['n'])
            big_on_times=np.array(['n'])

        if small_always_off == False and small_always_on == False and small_both_off == False:
            small_on_times= X.on_times[0]+small_times
            small_off_times=X.off_times[0]+small_times

        if big_always_off == False and big_always_on == False and big_both_off == False:
            big_on_times=Y.on_times[0]+big_times
            big_off_times=Y.off_times[0]+big_times

        # times_of_interest=np.concatenate((zero,small_on_times,big_on_times,small_off_times,big_off_times))
        # sorted_toi=np.sort(times_of_interest)

    else:
        #zero=np.arange(0,1,1)
        if len(X.off_times) <=0:
            small_times= np.arange(0,((3*X.period)),Y.period)
            big_times= np.arange(0,((3*X.period)),X.period)
        else:
            small_times= np.arange(0,((3*X.period)+X.off_times[0]),Y.period)
            big_times= np.arange(0,((3*X.period)+X.off_times[0]),X.period)


        if len(Y.on_times)<=0 and len(Y.off_times)>0:
            small_always_off = True
        if len(Y.on_times)>0 and len(Y.off_times)<=0:
            small_always_on = True
        if len(Y.on_times)<=0 and len(Y.off_times)<=0:
            small_both_off = True

        if len(X.on_times)<=0 and len(X.off_times)>0:
            big_always_off = True
        if len(X.on_times)>0 and len(X.off_times)<=0:
            big_always_on = True
        if len(X.on_times)<=0 and len(X.off_times)<=0:
            big_both_off = True

        if small_always_off == True and small_both_off == False:
            small_off_times=Y.off_times[0]+small_times  
            small_on_times=np.array(['n'])

        if small_always_on == True:
            small_off_times=np.array(['n'])
            small_on_times = Y.on_times[0]+small_times

        if small_both_off == True:
            small_off_times=np.array(['n'])
            small_on_times=np.array(['n'])
        
        if big_always_off == True:
            big_off_times=X.off_times[0]+big_times
            big_on_times=np.array('n')

        if big_always_on == True:
            big_off_times=np.array(['n'])
            big_on_times=X.on_times[0]+big_times

        if big_both_off == True :
            big_off_times=np.array(['n'])
            big_on_times=np.array(['n'])
            

        if small_always_off == False and small_always_on == False and small_both_off == False:
            small_on_times= Y.on_times[0]+small_times
            small_off_times= Y.off_times[0]+small_times

        if big_always_off == False and big_always_on == False and big_both_off == False:
            big_on_times=X.on_times[0]+big_times
            big_off_times=X.off_times[0]+big_times

        # times_of_interest=np.concatenate((zero,small_on_times,big_on_times,small_off_times,big_off_times))
        # sorted_toi=np.sort(times_of_interest)

    toi_list=[]
    toi_list.append(0)
    # print("big always on is:", big_always_on)
    # print("big always off is:", big_always_off)
    # print("big both off is:", big_both_off)
    # print("small always on is:", small_always_on)
    # print("small always off is:", small_always_off)
    # print("small both off is:", small_both_off)
    # print(X,Y)
    #print("big on times are:", big_on_times)

    for i in small_on_times:
        if i in toi_list or i =='n':
            continue
        else:
            toi_list.append(i)

    for i in small_off_times:
        if i in toi_list or i =='n':
            continue
        else:
            toi_list.append(i)

    for i in big_on_times:
        if i in toi_list or i =='n':
            continue
        else:
            toi_list.append(i)

    for i in big_off_times:
        if i in toi_list or i == 'n':
            continue
        else:
            toi_list.append(i)

    if toi_list==[0]:
        toi_list.append(np.max([X.period,Y.period]))

    toi_list = np.array(toi_list)
    toi_list = np.sort(toi_list)
    small_state=False
    big_state=False
    insync=False
    #print(toi_list)
    for i in range(len(toi_list)):
        if toi_list[i] in small_on_times:
            small_state =True
        if toi_list[i] in big_on_times:
            big_state = True
        if toi_list[i] in small_off_times:
            small_state = False
        if toi_list[i] in big_off_times:
            big_state = False
        
        if big_state == small_state and insync == False:
            change_events+=1
            insync = True
        elif big_state ==small_state and insync == True:
            #print(i)
            #print(big_state)
            #print(small_state)
            overlap_count+=toi_list[i] - toi_list[i-1]
            if i == len(toi_list)-1:
                sub_strings.append(overlap_count)

        elif big_state != small_state and insync == True:
            change_events+=1
            insync = False
            overlap_count+=toi_list[i] - toi_list[i-1]
            sub_strings.append(overlap_count)
            overlap_count=0
        elif big_state !=small_state and insync == False:
            continue
    #print(toi_list)
    # print(change_events)
    answer = np.max(sub_strings) / np.max([X.period,Y.period])
    if answer >1:
        answer = 1
    return(answer)

def energy_cost(X):
    """ where X is a timeseries.
        Where max_time is the maximum 
    determine the power value assigned to a generated sequence based on the number and length of pulses produced"""
    pulse_duration=0
    if X.on_times==[] and X.off_times==[]:
        energy_value=0
    elif X.on_times==[]:
        energy_value=0 ### if signal never pulses, no energy used
    elif X.off_times==[]:
        energy_value=1
    else:
        for i in range(len(X.on_times)):
            pulse_duration+= (X.off_times[i]-X.on_times[i])
        energy_value=pulse_duration/X.period
    
    if energy_value>1:
        energy_value=1
    return energy_value

def signal_cost(X,background,ws,we):
    """"
    Where X is the generated signal time series
    background is a list of Time Series that represent the background pulsars
     ws is the weight assigned to the similarity value
    we is the weight assigned to the energy value
    """
    similarity=0 ### initialize the similarity value of the generated signal
    for i in background:
        #print('similarity background index is', i)
        similarity+= eval_overlap(X,i)  ### since eval_overlap gives same value as overlap_cost but faster
        #print(similarity)
    #print("similarities finished processing")
    
    sim=similarity/len(background)
    cost= 100*(ws*sim +we*energy_cost(X))
    return cost

def new_pulsarsim(period,w50,meanflux):
    """ Generate a simulated pulse profile based on the given period, width, and mean flux.
    :param period: The pulse period in seconds.
    :param w50: The width of the pulse at half maximum in milliseconds.
    :param meanflux: The mean flux of the pulse.
    :return: A tuple containing the time array and the pulse profile.
    """

    ### Convert period from seconds to milliseconds for consistency
    period=period*1000

    ### use the width at half maximum to calculate the standard deviation
    ### of a Gaussian pulse profile
    sigma=w50/(2*np.sqrt(2*np.log(2)))

    ### Calculate the amplitude of the pulse profile based on the mean flux and period
    ### The amplitude is calculated using the formula for the area under a Gaussian curve
    amplitude=meanflux*period/(np.sqrt(2*np.pi)*sigma)

    ### Generate the time array for the pulse profile
    time=np.linspace(0,period,1000)

    ### Generate the pulse profile using a Gaussian function
    ### The pulse profile is centered at t0, which is half the period
    t0=period/2
    pulse_profile=amplitude*np.exp(-0.5*((time-t0)/sigma)**2)

    ### Return the time array and the pulse profile
    return time, pulse_profile  

def new_pulsar_modeller(Center,fluxcutoff,x,y,z,r):
    """
    Generate a list of TimeSeries instances representing background pulsars based on user input parameters.
    :param Center: The frequency center (400 or 1400 MHz).
    :param fluxcutoff: The flux cutoff value.
    :param x: The x-coordinate for the pulsar search.
    :param y: The y-coordinate for the pulsar search.
    :param z: The z-coordinate for the pulsar search.
    :param r: The radius for the pulsar search.
    :return: A tuple containing a list of TimeSeries instances representing the background pulsars, and a list of their names
    """
    background=[]
    pulsef=[]
    points=[]
    names=[]
    ### Below, the function will choose between the two frequencies available in the database given User Input
    if Center == 400:
        Center== 'S400'
    if Center == 1400:
        Center = 'S1400'
    ### psrqpy is used to query the ATNF database for pulsars based on the user input parameters
    d= psrqpy.QueryATNF(params=["JNAME",'W50',Center, "P0", "F0", "DecJ","DecJD","RaJ","RaJD","GL","GB","ZZ","XX","YY"])
    data=d.catalogue
    ### Below, the data is filtered based on the user input parameters
    database=data[(data[Center] > fluxcutoff) &(data["ZZ"]<=z+r) & (data["ZZ"]>=z-r) & (data["XX"]<=x+r) & (data["XX"]>=x-r) & (data["YY"]<=y+r) & (data["YY"]>=y-r) ]
    
    for i in range(len(database[Center])):

        ####below the pulsar info is extracted from database for each pulsar
        pulsename= database["JNAME"].iloc[i]
        names.append(pulsename)
        periodz= database["P0"].iloc[i]
        frequency= database['F0'].iloc[i]
        ### new_pulsarsim will generate a simulated pulse profile based on extracted values
        avgflux=database[Center].iloc[i]
        point,pulse=new_pulsarsim(periodz,database['W50'].iloc[i],avgflux)

        ### The pulse profile is then processed to create a TimeSeries instance
        t_factor=1
        on_timer=[]
        off_timer=[]

        ### The pulse profile is processed to determine the on and off times based on the flux cutoff
        ### and the flux value at each point in the pulse profile
        for i in range(len(pulse)):
            if i == len(pulse)-1:

                ## If the pulse value of last point is below the flux cutoff and the previous pulse value was above the cutoff,
                ## then the current point is added to the off_timer
                if pulse[i] < fluxcutoff and pulse[i-1]>= fluxcutoff:
                    if np.round((point[i]*t_factor),3) in off_timer:
                        continue
                    else:
                        off_timer.append(np.round((point[i]*t_factor),3))

                ## If the pulse value of last point is above the flux cutoff and the previous pulse value was below the cutoff,
                ## then the current point is added to the on_timer
                elif pulse[i]> fluxcutoff and pulse[i-1] <fluxcutoff:
                    if np.round((point[i]*t_factor),3) in on_timer:
                        continue
                    else:
                        on_timer.append(np.round((point[i]*t_factor),3))

            #### if the pulse value of the current point is below the flux cutoff and the next pulse value is above the cutoff,
            ### then the next point is added to the on_timer if not already present
            elif pulse[i] < fluxcutoff and pulse[i+1]>= fluxcutoff:
                if np.round((point[i+1]*t_factor),3) in on_timer:
                    continue
                else:
                    on_timer.append(np.round((point[i+1]*t_factor),3))

            ### if the pulse value of the current point is above the flux cutoff and the next pulse value is below the cutoff,
            ### then the next point is added to the off_timer if not already present
            elif pulse[i]> fluxcutoff and pulse[i+1] <fluxcutoff:
                if np.round((point[i+1]*t_factor),3) in off_timer:
                    continue
                else:
                    off_timer.append(np.round((point[i+1]*t_factor),3))

            ### if the pulse value of the first point is below the flux cutoff and the next pulse value is above the cutoff,
            ### then the next point is added to the on_timer if not already present
            elif i ==0 and pulse[i] < fluxcutoff and pulse[i+1]>= fluxcutoff:
                on_timer.append(np.round((point[i+1]*t_factor),3))

            ### if the pulse value of the first point is above the flux cutoff and the next pulse value is also above the cutoff,
            ### then the next point is added to the on_timer if not already present
            elif i ==0 and pulse[i] > fluxcutoff and pulse[i+1] > fluxcutoff:
                on_timer.append(np.round((point[i]*t_factor),3))

        ### the pulsar is then created as a TimeSeries instance with the on and off times        
        pulsar=TimeSeries(on_times=on_timer, off_times= off_timer, period = np.round(periodz*1000,3))  
        
        ### Uf the pulsar has no off times, then the off time is set to the period of the pulsar
        ### This is done to ensure the pulsar has a valid structure and represents a pulsar that is always on
        if len(pulsar.on_times)>0 and pulsar.off_times==[]:
            pulsar.off_times.append(np.round(periodz*1000,3))

        ### The pulsar is then added to the background list    
        background.append(pulsar)

        ### The pulse profile and point values are also added to their respective lists
        ### for testing purposes if needed
        #pulsef.append(pulse)
        #points.append(point)

    #return(background,pulsef,points)
    return background,names


def process_combination(params):
    """ Process a single combination of parameters to generate a TimeSeries and calculate its cost.
    :param params: A tuple containing the parameters (periodz, pulse_fract, background, ws, we).
    :return: A tuple containing the generated TimeSeries and its cost.
    """
    periodz, pulse_fract, background, ws, we = params
    ### Generate the TimeSeries based on the given parameters

    generated = TimeSeries(on_times=[0], off_times=[pulse_fract * periodz], period=periodz)

    ### Calculate the cost of the generated TimeSeries
    prospect_cost = signal_cost(generated, background, ws, we)

    return generated, prospect_cost

def exp_signal_modeller(background, ws, we,min_period, large_period, periodstep, pulsestep,pulsemin,pulsemax):
    """ Generate a champion signal based on the given parameters and background pulsars.
    :param background: List of TimeSeries instances representing background pulsars.
    :param ws: Weight assigned to the similarity value.
    :param we: Weight assigned to the energy value.
    :param min_period: Minimum pulse period in milliseconds.
    :param large_period: Maximum pulse period in milliseconds.
    :param periodstep: Step size for pulse periods in milliseconds.
    :param pulsestep: Step size for absolute duty cycles.
    :param pulsemin: Minimum absolute duty cycle.
    :param pulsemax: Maximum absolute duty cycle.
    :return: A tuple containing the champion signal, its cost, and lists of all generated signals and their costs.
    """
    ### Create a range of pulse periods and absolute duty cycles
    float_array = np.arange(min_period, large_period + periodstep, periodstep, dtype=float) # Create pulse period values
    fraction_array = np.arange(pulsemin, pulsemax+pulsestep, pulsestep, dtype=float) # Create absolute duty cycle values

    ### Prepare the list of combinations of periods and absolute duty cycles
    combinations = [(periodz, pulse_fract, background, ws, we) for periodz in float_array for pulse_fract in fraction_array]
    
    signals = []
    prospect_costs = []

    ### Create a multiprocessing pool
    num_processes = mp.cpu_count()  # Or set a specific number of processes
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_combination, combinations), total=len(combinations), desc="Generating signals"))

    ### Separate the results into signals and prospect_costs
    signals, prospect_costs = zip(*results)

    ### Identify the champion signal and its cost as the minimum of all prospect costs
    champion_cost = np.min(prospect_costs)
    champion_signal = signals[np.argmin(prospect_costs)]
    ### Return the champion signal, its cost, and the lists of all generated signals and their costs
    return champion_signal, champion_cost, signals, prospect_costs


def heat_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,weightname):
    """    Generate a heat map and contour map of costs for different pulse periods and absolute duty cycles.
    :param sorted_cs: List of sorted costs of all signals generated when finding champion signal
    :param min_period: Minimum pulse period in milliseconds.
    :param large_period: Maximum pulse period in milliseconds.
    :param periodstep: Step size for pulse periods in milliseconds.
    :param pulsestep: Step size for absolute duty cycles.
    :param absmin: Minimum absolute duty cycle.
    :param absmax: Maximum absolute duty cycle.
    :param weightname: Name of the weight used for the cost calculation, used for saving the heatmap.
    """
    
    ### Create a grid of pulse periods and absolute duty cycles
    float_array = np.round(np.arange(min_period, large_period + periodstep, periodstep, dtype=float),decimals=2) ## create pulse period values for heatmap
    fraction_array = np.round(np.arange(absmin, absmax+pulsestep, pulsestep, dtype=float),decimals=2) ## create absolute duty cycle values for heatmap

    ### Ensure the sorted costs are in the correct shape for the heatmap
    shaped_costs=np.array(sorted_cs).reshape(len(float_array),len(fraction_array))

    ### Create a DataFrame for the costs
    costs_df = pd.DataFrame(shaped_costs, index=float_array, columns=fraction_array)

    ### Create the heatmap
    plt.figure(figsize=(12, 10))

    sns.heatmap(costs_df, cmap='magma', cbar_kws={'label': 'Cost '}, linewidths=0)
    
    ###Add labels and title
    plt.xlabel('Absolute Duty Cycle', fontsize=20)
    plt.ylabel('Pulse Periods (ms)', fontsize=20)
    plt.title(f'Heat Map of Costs (Ws={weightname})', fontsize=20, fontweight='bold')

    ### save the heatmap
    plt.savefig(f'Heatmap_{weightname}.png')
    plt.clf()
    #plt.show()

    ### Create a contour map
def contour_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,weightname):
    """ Generate a contour map of costs for different pulse periods and absolute duty cycles.
    
    :param sorted_cs: List of sorted costs of all signals generated when finding champion signal
    :param min_period: Minimum pulse period in milliseconds.
    :param large_period: Maximum pulse period in milliseconds.
    :param periodstep: Step size for pulse periods in milliseconds.
    :param pulsestep: Step size for absolute duty cycles.
    :param absmin: Minimum absolute duty cycle.
    :param absmax: Maximum absolute duty cycle. 
    :param weightname: Name of the weight used for the cost calculation, used for saving the contour map.
    """
    ### Create a grid of pulse periods and absolute duty cycles
    float_array = np.round(np.arange(min_period, large_period + periodstep, periodstep, dtype=float),decimals=2) ## create pulse period values for contour map
    fraction_array = np.round(np.arange(absmin, absmax+pulsestep, pulsestep, dtype=float),decimals=2) ## create absolute duty cycle values for contour map

    ### Ensure the sorted costs are in the correct shape for the contour map
    shaped_costs=np.array(sorted_cs).reshape(len(float_array),len(fraction_array))
    
    plt.figure(figsize=(12, 10))

    num_levels = 10 ## 10 is chosen to determine number of contour levels.

    # Create contour levels based on the minimum and maximum costs
    levels = np.linspace(np.min(shaped_costs), np.max(shaped_costs), num_levels)
    plt.contour(fraction_array,float_array,shaped_costs,colors='black',levels=levels)
    plt.contourf(fraction_array,float_array,shaped_costs,cmap='magma', levels=levels)

    # Add color bar and labels
    cbar=plt.colorbar()
    cbar.set_label('Cost',fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Absolute Duty Cycle',fontsize=20)
    plt.ylabel('Pulse Periods (ms)',fontsize=20)
    plt.title(f'Contour Map of Costs (Ws={weightname})', fontsize=20,fontweight='bold')
    plt.yticks(np.round(np.arange(min_period, large_period + periodstep, periodstep*10),2))
    plt.xticks(np.round(np.arange(absmin, absmax, pulsestep*10),2))
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.tight_layout()

    ### Save the contour map
    plt.savefig(f'ContourMap_{weightname}.png')
    plt.clf()
    #plt.show()

if __name__ == "__main__":
    print("Welcome to the Pulsar Modeller! \n")

    operator=str(input("What would you like to generate? [optimized pulsar, 2d star map, 3d star map , sequence figure, cost distribution graph]: ")).lower()
    if operator == 'optimized pulsar':
        backgroundcheck=str(input("\nDo you have background pulsars already? (yes or no): ")).lower()

        if backgroundcheck == 'no':
            ### Prompt the user for input parameters to generate a background pulsar database
            print("Please enter the following parameters:")
            Center = int(input("Frequency center (400 or 1400 MHz (Do not include units)): "))
            fluxcutoff = float(input("Flux cutoff (mJy): "))
            x = float(input("X-coordinate for pulsar search (kpc): "))
            y = float(input("Y-coordinate for pulsar search (kpc): "))
            z = float(input("Z-coordinate for pulsar search (kpc): "))
            r = float(input("Radius for pulsar search (kpc): "))
            backname= str(input("what should the file name of the background pulsars be? (without .txt): "))
            background, names = new_pulsar_modeller(Center, fluxcutoff, x, y, z, r)
            print(f"Found {len(background)} pulsars in the database.")

            with open(f"./{backname}.txt", 'w') as file:
                for i,item in enumerate(background):
                    file.write(str(item))
                    if i < len(background) - 1:
                        file.write(", ")
            print(f"Background pulsars saved to {backname}.txt")

            with open(f"./{backname}_names.txt", 'w') as file:
                for i,item in enumerate(names):
                    file.write(str(item))
                    if i < len(names) - 1:
                        file.write(", ")
            print(f"Background pulsars names saved to {backname}_names.txt")

        elif backgroundcheck == 'yes':
            backname = str(input("what is the file name of the background pulsars? (without .txt): "))
            with open(f"./{backname}.txt", 'r') as file:
                data = file.read()
                background = eval(data)
            print(f"Loaded {len(background)} pulsars from {backname}.txt")

        ### Prompt the user for additional parameters to generate a champion signal(s)
        print("\nNow, please enter the parameters for generating the optimized Firefly signal(s):\n")
        howmanychamps=str(input("Would you like to test multiple different similarity weights? (yes or no): ")).lower()
        if howmanychamps == 'yes':
            ### Prompt the user for multiple similarity weights they want to test
            raw = input("Enter the similarity weights (values between 0 and 1): ")
            wslist = [float(item.strip()) for item in raw.split(",")]
        elif howmanychamps == 'no':
            ### Prompt the user for a single similarity weight
            ws= float(input("Enter the similarity weight (between 0 and 1): "))
            we= 1.0 - ws
        resolution = int(input("Enter the resolution of the parameter space to search (number of steps between min and max period and absolute duty cycle): "))

        saveall=str(input("Would you like to save the generated signal and cost landscape(s) to a file? (yes or no): ")).lower()
        contourheat=str(input("Would you like to generate a contour or heatmap or both? (contour or heatmap or both?): ")).lower()

        ## Calculate the minimum and maximum pulse period and absolute duty cycle based on the background pulsars
        ## and set the step sizes for pulse period and absolute duty cycle
        ## This is done to ensure the generated signals are within the normal distribution of the background pulsars
        perioded=[]
        abscyc=[]
        for i in background:
            perioded.append(i.period)
            abscyc.append(energy_cost(i))

        ### Calculate the minimum and maximum pulse period and absolute duty cycle using normal distribution
        min_period=np.mean(perioded)-2* np.std(perioded) 
        large_period=np.mean(perioded)+2* np.std(perioded)
        absmin=np.mean(abscyc)-2* np.std(abscyc)
        absmax=np.mean(abscyc)+2* np.std(abscyc)

        ### Calculate the minimum and maximum pulse period and absolute duty cycle to scale in on anomalies.
        # min_period = 2
        # large_period = 10
        ### Ensure the minimum and maximum values are within valid ranges
        if min_period<0:
            min_period=10
        if absmin<0:
            absmin=0
        if absmax>1:
            absmax=1
        
        ### Set the step sizes for absolute duty cycle and pulse period by determing number of steps
        periodstep = round((large_period-min_period)/resolution,3)  # can change the step size by changing the resolution
        pulsestep = round((absmax-absmin)/resolution,3)  # can change the step size by changing the resolution

        #### Ensure the minimum and maximum values are within valid ranges
        if min_period==0:
            min_period=periodstep
        if absmin==0:
            absmin=pulsestep

        ### now to determine whether the user wants to test multiple ws weights or just one for the firefly signal
        if howmanychamps == 'yes':
            ### Prompt the user for multiple similarity weights they want to test
            # raw = input("Enter the similarity weights (values between 0 and 1): ")
            # wslist = [float(item.strip()) for item in raw.split(",")]
            
            ### create the firefly signal for a ws value of 1.0 to use as baseline for other ws values
            champion_signal, champion_cost, all_signals, all_costs = exp_signal_modeller(background, 1.0,0.0,min_period, large_period, periodstep, pulsestep,absmin,absmax)
            
            #### here is where the signal and cost landscape is sorted based on period and absolute duty cycle
            timeseries_withcosts=list(zip(all_signals,all_costs))
            sorted_timeseries_withcosts=sorted(timeseries_withcosts,key=lambda pair: (pair[0].period,pair[0].off_times))
            sorted_times,sortedcs=zip(*sorted_timeseries_withcosts)
            sorted_ts=list(sorted_times)
            sorted_cs=list(sortedcs)
            ### if the user wants to save the generated signal landscape do so, 
            ### signal landscape should be the same for all ws values
            if saveall == 'yes':
                with open(f"./fireflylandscape_all_signals.txt", 'w') as file:
                    for i,item in enumerate(sorted_ts):
                        file.write(str(item))
                        if i < len(sorted_ts) - 1:
                            file.write(", ")
            ### initialize lists to store the firefly costs and signals for each ws value
            firefly_costs=[]
            firefly_signals=[]
            ### Loop through each weight in the wslist and calculate the firefly signal and cost
            for weight in wslist:
                templist=[]
                for i in range(len(sorted_cs)):
                    excost= weight*sorted_cs[i]+ (1-weight)*100*energy_cost(sorted_ts[i])
                    templist.append(excost)
                firefly_costs.append(np.min(templist))
                firefly_signals.append(sorted_ts[np.argmin(templist)])
                print(f"when ws is {weight} the firelfy signal cost is {np.min(templist):.2f}%")
                print(f"when ws is {weight} the firelfy signal is {sorted_ts[np.argmin(templist)]}")

                ### if the user wants to save the generated signal and cost landscape for each ws value, do so
                if saveall == 'yes':
                    with open(f"./{weight}_all_costs.txt", 'w') as file:
                        for i,item in enumerate(templist):
                            file.write(str(item))
                            if i < len(templist) - 1:
                                file.write(", ")
                ### If the user wants to generate a contour or heatmap or both, do so
                if contourheat == 'contour':
                    contour_map(templist,min_period, large_period, periodstep,pulsestep,absmin,absmax,weight)
                elif contourheat == 'heatmap':
                    heat_map(templist,min_period, large_period, periodstep,pulsestep,absmin,absmax,weight)
                elif contourheat == 'both':
                    contour_map(templist,min_period, large_period, periodstep,pulsestep,absmin,absmax,weight)
                    heat_map(templist,min_period, large_period, periodstep,pulsestep,absmin,absmax,weight)
            
            ### Save the firefly costs and signals for each ws value to respective files
            with open(f"./multiple_ws_all_costs.txt", 'w') as file:
                for i,item in enumerate(firefly_costs):
                    file.write(str(item)+" when ws is "+str(wslist[i]))
                    if i < len(firefly_costs) - 1:
                        file.write(", ")
            with open(f"./multiple_ws_all_signals.txt", 'w') as file:
                for i,item in enumerate(firefly_signals):
                    file.write(str(item)+" when ws is "+str(wslist[i]))
                    if i < len(firefly_signals) - 1:
                        file.write(", ")
            
            ### Print confirmation messages for the user and file names of saved data
            print("All generated optimizedFirefly signals and Firefly costs saved to multiple_ws_all_signals.txt and multiple_ws_all_costs.txt")
            if saveall == 'yes':
                print("All generated signals for landscape saved to file named fireflylandscape_all_signals.txt")
                print("All generated costs for landscape saved to files with the ws value in the name. E.g.  0.5_all_costs.txt")

            if contourheat == 'contour':
                print("Contour map generated and saved for each ws value to files named ContourMap_<ws_value>.png")
            elif contourheat == 'heatmap':
                print("Heat map generated and saved for each ws value to files named HeatMap_<ws_value>.png")
            elif contourheat == 'both':
                print("Contour map generated and saved for each ws value to files named ContourMap_<ws_value>.png")
                print("Heat map generated and saved for each ws value to files named HeatMap_<ws_value>.png")
        
        ### if user does not want to test multiple similarity weights
        elif howmanychamps == 'no':
            ### Prompt the user for a single similarity weight
            # ws= float(input("Enter the similarity weight (between 0 and 1): "))
            # we= 1.0 - ws
            champion_signal, champion_cost, all_signals, all_costs = exp_signal_modeller(background, ws, we,min_period, large_period, periodstep, pulsestep,absmin,absmax)
            
            ### Print the optimized Firefly signal and cost
            print ("when ws is ",ws)
            print("Optimized Firefly signal generated: ", champion_signal)
            print("Optimized Firefly signal cost: ", champion_cost)

            ### sort the generated signals and costs based on period and absolute duty cycle
            timeseries_withcosts=list(zip(all_signals,all_costs))
            sorted_timeseries_withcosts=sorted(timeseries_withcosts,key=lambda pair: (pair[0].period,pair[0].off_times))
            sorted_times,sortedcs=zip(*sorted_timeseries_withcosts)
            sorted_ts=list(sorted_times)
            sorted_cs=list(sortedcs)

            #### If the user wants to save the generated signal and cost landscape, do so
            if saveall == 'yes':
                
                with open(f"./{ws}_all_signals.txt", 'w') as file:
                    for i,item in enumerate(sorted_ts):
                        file.write(str(item))
                        if i < len(sorted_ts) - 1:
                            file.write(", ")

                with open(f"./{ws}_all_costs.txt", 'w') as file:
                    for i,item in enumerate(sorted_cs):
                        file.write(str(item))
                        if i < len(sorted_cs) - 1:
                            file.write(", ")
                
                ### Print confirmation messages for the user and file names of saved data
                print(f"All generated signals for landscape saved to {ws}_all_signals.txt")
                print(f"All generated costs for landscape saved to {ws}_all_costs.txt")
            
            ### If the user wants to generate a contour or heatmap or both, do so
            if contourheat == 'contour':
                contour_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,ws)
                print("Contour map generated and saved for each ws value to files named ContourMap_<ws_value>.png")
            elif contourheat == 'heatmap':
                heat_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,ws)
                print("Heat map generated and saved for each ws value to files named HeatMap_<ws_value>.png")
            elif contourheat == 'both':
                contour_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,ws)
                heat_map(sorted_cs,min_period, large_period, periodstep,pulsestep,absmin,absmax,ws)
                print("Contour map generated and saved for each ws value to files named ContourMap_<ws_value>.png")
                print("Heat map generated and saved for each ws value to files named HeatMap_<ws_value>.png")

    elif operator == '2d star map':
        # Wrap in Astropy CartesianRepresentation (in kpc)
        print("\nPlease enter the following parameters to populate pulsar background:\n")
        
        # Define parameters to identify pulsars to map
        Center = int(input("Frequency center (400 or 1400 MHz (Do not include units)): "))
        if Center == 400:
            Center = 'S400'
        elif Center == 1400:
            Center = 'S1400'
        fluxcutoff = float(input("Flux cutoff (mJy): "))
        x = float(input("X-coordinate for pulsar search (kpc): "))
        y = float(input("Y-coordinate for pulsar search (kpc): "))
        z = float(input("Z-coordinate for pulsar search (kpc): "))
        r = float(input("Radius for pulsar search (kpc): "))

        # Access ATNF database to find all pulsars matching assigned parameters
        d= psrqpy.QueryATNF(params=["JNAME",'W50',Center, "P0", "F0", "DecJ","DecJD","RaJ","RaJD","GL","GB","ZZ","XX","YY"])
        data=d.catalogue
        database=data[(data[Center] > fluxcutoff) &(data["ZZ"]<=z+r) & (data["ZZ"]>=z-r) & (data["XX"]<=x+r) & (data["XX"]>=x-r) & (data["YY"]<=y+r) & (data["YY"]>=y-r) ]
        wumbo=[]
        #print(database["XX"][1])
        for i in range(len(database)):
            wumbo.append([database["XX"].iloc[i],database["YY"].iloc[i],database["ZZ"].iloc[i]])
        #print(wumbo)
            

        # ----------------------------
        coords_kpc = np.array(wumbo)
        
        pulsars_cart = CartesianRepresentation(
            coords_kpc[:, 0] * u.kpc,
            coords_kpc[:, 1] * u.kpc,
            coords_kpc[:, 2] * u.kpc
        )

        # Convert to Galactic coordinates
        pulsars_gal = SkyCoord(pulsars_cart, frame='galactic')

        # Extract Galactic longitude (l) and latitude (b) in radians for Mollweide
        l_rad = pulsars_gal.l.wrap_at(180 * u.deg).radian
        b_rad = pulsars_gal.b.radian

        # Create figure with Mollweide projection
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='mollweide')

        # Plot pulsars
        ax.scatter(l_rad, b_rad, c='gold',s=120, linewidth=1.2, edgecolor='k', label='Pulsars')

        # Plot Galactic Center at (l=0, b=0)
        ax.scatter(0, 0, c='red', s=140, marker='*', label='Galactic Center')

        # Plot Galactic Plane (b=0 line)
        plane_l = np.linspace(-180, 180, 500) * u.deg
        plane_b = np.zeros_like(plane_l.value) * u.deg
        ax.plot(plane_l.to(u.rad), plane_b.to(u.rad), color='gray', alpha=0.5,lw=3.0, label='Galactic Plane')


        # Style
        ax.set_xticklabels(['150°','120°','90°','60°','30°','0°','330°','300°','270°','240°','210°'],fontsize=28)
        ax.set_yticklabels(['-75°','-60°','-45°','-30°','-15°','0°', '15°', '30°', '45°', '60°', '75°'], fontsize=28)
        ax.set_xlabel('Galactic Longitude (l)',fontsize=28)
        ax.set_ylabel('Galactic Latitude (b)',fontsize=28)
        ax.set_title('Pulsars Map (Mollweide Projection)',fontsize=28,pad=20, fontweight='bold')

        # Legend
        ax.legend(loc='lower left',fontsize=28,ncols=3,bbox_to_anchor=(-0.1, -0.20))

        plt.grid(True, color='lightgray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    elif operator == '3d star map':
        # Wrap in Astropy CartesianRepresentation (in kpc)
        print("\nPlease enter the following parameters to populate pulsar background:\n")
        
        # Define parameters to identify pulsars to map
        Center = int(input("Frequency center (400 or 1400 MHz (Do not include units)): "))
        if Center == 400:
            Center = 'S400'
        elif Center == 1400:
            Center = 'S1400'
        fluxcutoff = float(input("Flux cutoff (mJy): "))
        x = float(input("X-coordinate for pulsar search (kpc): "))
        y = float(input("Y-coordinate for pulsar search (kpc): "))
        z = float(input("Z-coordinate for pulsar search (kpc): "))
        r = float(input("Radius for pulsar search (kpc): "))
    
        d= psrqpy.QueryATNF(params=["JNAME",'W50',Center, "P0", "F0", "DecJ","DecJD","RaJ","RaJD","GL","GB","ZZ","XX","YY"])
        data=d.catalogue
        database=data[(data[Center] > fluxcutoff) &(data["ZZ"]<=z+r) & (data["ZZ"]>=z-r) & (data["XX"]<=x+r) & (data["XX"]>=x-r) & (data["YY"]<=y+r) & (data["YY"]>=y-r) ]
        wumbo=[]
        #print(database["XX"][1])
        for i in range(len(database)):
            wumbo.append([database["XX"].iloc[i],database["YY"].iloc[i],database["ZZ"].iloc[i]])
        #print(wumbo)

        coords_kpc = np.array(wumbo)
        # Wrap in Astropy CartesianRepresentation (in kpc)
        pulsars = CartesianRepresentation(
            coords_kpc[:, 0] * u.kpc,
            coords_kpc[:, 1] * u.kpc,
            coords_kpc[:, 2] * u.kpc
        )

        # Create figure
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot pulsars 
        ax.scatter(
            pulsars.x.value, pulsars.y.value, pulsars.z.value,
            c='gold', s=50, edgecolor='k', label='Pulsars'
        )

        # Plot Earth
        ax.scatter(0, 0, 0, c='blue', s=100, label='Earth')

        # Plot Galactic Center (approx. +8 kpc in X direction)
        gc = CartesianRepresentation(8.0 * u.kpc, 0 * u.kpc, 0 * u.kpc)
        ax.scatter(gc.x.value, gc.y.value, gc.z.value, c='red', s=150, marker='*', label='Galactic Center')

        # Galactic plane centered on Galactic Center
        plane_radius = 26.8 ## rough radius of galaxy
        theta = np.linspace(0, 2*np.pi, 300)
        plane_x = gc.x.value + plane_radius * np.cos(theta)
        plane_y = gc.y.value + plane_radius * np.sin(theta)
        plane_z = np.zeros_like(theta) + gc.z.value

        # Outline of galactic plane
        ax.plot(plane_x, plane_y, plane_z, color='gray', alpha=0.4, linewidth=1.5)
        # Semi-transparent fill of galactic plane
        ax.plot_trisurf(plane_x, plane_y, plane_z, color='gray', alpha=0.08, edgecolor='none')
        # Empty patch for legend
        galactic_plane_patch = Patch(facecolor='gray', alpha=0.3, label='Galactic Plane')


        # Labels and style
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')
        ax.set_zlabel('Z (kpc)')
        ax.set_title('3D Pulsar Map (Astropy Cartesian Representation)')

        # Equal aspect ratio
        max_range = np.array([
            (pulsars.x.max() - pulsars.x.min()).to_value(u.kpc),
            (pulsars.y.max() - pulsars.y.min()).to_value(u.kpc),
            (pulsars.z.max() - pulsars.z.min()).to_value(u.kpc)
        ]).max() / 2.0

        mid_x = ((pulsars.x.max() + pulsars.x.min()) / 2.0).to_value(u.kpc)
        mid_y = ((pulsars.y.max() + pulsars.y.min()) / 2.0).to_value(u.kpc)
        mid_z = ((pulsars.z.max() + pulsars.z.min()) / 2.0).to_value(u.kpc)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Legend 
        ax.legend(handles=ax.get_legend_handles_labels()[0] + [galactic_plane_patch])


        plt.show()

    elif operator == 'sequence figure':
        print("\nPlease enter the following information to generate the sequence figure:\n")

        # Define parameters to identify pulsars to map
        weights = input("Enter the similarity weights of the artificial signals to be graphed (values between 0 and 1): ")
        wslist = [float(item.strip()) for item in weights.split(",")]
        artsignals=input("enter the artificial signals to be graphed (seperate with - please): ")
        signallist=[eval(item.strip()) for item in artsignals.split("-")]
        print(signallist)
        print("\nPlease enter the following parameters to create pulsar background pulses:\n")
        # Define parameters to identify pulsars to map
        Center = int(input("Frequency center (400 or 1400 MHz (Do not include units)): "))
        fluxcutoff = float(input("Flux cutoff (mJy): "))
        x = float(input("X-coordinate for pulsar search (kpc): "))
        y = float(input("Y-coordinate for pulsar search (kpc): "))
        z = float(input("Z-coordinate for pulsar search (kpc): "))
        r = float(input("Radius for pulsar search (kpc): "))
        backnumber=int(input("How many background pulsars would you like to display on the figure?: "))
        
        # Generate background pulsars
        background, names = new_pulsar_modeller(Center, fluxcutoff, x, y, z, r)
        
        # Find the max period and offset for bounds of figure
        allpers=[]
        offsets=[]
        for i in range(backnumber):
            allpers.append(background[i].period)
            offsets.append(background[i].on_times[0])
        for i in range(len(signallist)):
            allpers.append(signallist[i].period)
            offsets.append(signallist[i].on_times[0])

        max_period = max(allpers)
        max_offset = max(offsets)
        colours=['purple','blue','green','black','red','orange','cyan','yellow','pink','brown']
        
        
        # Plot the artificial signals above the background signals
        for i in range(len(signallist)):
            val=i+1+backnumber
            anchor=np.array([val,val])

            pulsest=signallist[i].on_times[0]
            pulseend=signallist[i].off_times[0]
            pulse=np.array([pulsest,pulseend])
            pulseperiod=signallist[i].period
            plt.plot(pulse,anchor, color=colours[i], linewidth=10,label='ws='+str(wslist[i]))
            for z in np.arange(0,max_period+max_offset,pulseperiod):
                pulse=pulse+pulseperiod
                plt.plot(pulse,anchor, color=colours[i], linewidth=10)

        # Plot the background signals
        h=0
        for i in range(len(background)):
            if h==backnumber:
                break
            if len(background[i].on_times)==0 or background[i].on_times[0] > max_period+max_offset:
                continue
            else:
                a=np.array([background[i].on_times,background[i].off_times])
                plt.plot(a,[i+1,i+1], color='grey', linewidth=10)
                for z in np.arange(0,max_period+max_offset,background[i].period):
                    a=a+z
                    plt.plot(a,[h+1,h+1], color='grey', linewidth=10)
                h+=1

        # Add a label for the background pulsars
        plt.plot(-1,0, color='grey',label='Background Pulsars', linewidth=10)
        #plt.xlim(-0.001,1.1*z3[1])

        # Label and Style
        plt.xlim(-1,max_period+max_offset)
        plt.ylim(0,backnumber+len(signallist)+1)
        plt.xlabel('Time (ms)',fontsize=26)
        plt.ylabel('Signal',fontsize=26)
        plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15),ncols = 5, fontsize=24)
        plt.yticks([])
        plt.tick_params(axis='x', labelsize=20)

        plt.title('Background Pulsar Signals and Generated Signals when Cutoff='+str(Center) + 'mJy',fontsize=24,fontweight='bold')
        plt.show()

    elif operator == 'cost distribution graph':
        print("\nPlease enter the following information to generate the cost distribution graph:\n")

        weights = input("Enter the similarity weights of the artificial signals to be graphed (values between 0 and 1) (max 4 weights): ")
        wslist = [float(item.strip()) for item in weights.split(",")]
        artcosts=input("enter the artificial signal costs: ")
        artcostlist = [float(item.strip()) for item in artcosts.split(",")]

        print("\nPlease enter the following parameters to create pulsar background pulses:\n")
        # Define parameters to identify pulsars to map
        Center = int(input("Frequency center (400 or 1400 MHz (Do not include units)): "))
        fluxcutoff = float(input("Flux cutoff (mJy): "))
        x = float(input("X-coordinate for pulsar search (kpc): "))
        y = float(input("Y-coordinate for pulsar search (kpc): "))
        z = float(input("Z-coordinate for pulsar search (kpc): "))
        r = float(input("Radius for pulsar search (kpc): "))

        background, names = new_pulsar_modeller(Center, fluxcutoff, x, y, z, r)
        
        print("\n")


        fig, axs = plt.subplots(1, len(wslist), figsize=(10, 6))


        for i in range(len(wslist)):
            data = []
            for x in background:
                # calculate the cost of each pulsar in background excluding itself
                expback=[item for item in background if item !=x]
                pcost=signal_cost(x,expback,wslist[i],1.0-wslist[i])
                data.append(pcost)
            # Calculate the mean and standard deviation of the background pulsar costs
            mu = np.mean(data)
            sigma = np.std(data)
            print(f"Mean: {mu:.2f}, Std Dev: {sigma:.2f} when ws={wslist[i]}")
            
            # Define the artificial signal cost at specified weight
            x_value = artcostlist[i]
            #Calculate the z-score for the artificial signal cost
            z_score = (x_value - mu) / sigma
            # print(z_score)

            #Calculate the percentile from the cumulative distribution function
            percentile = norm.cdf(z_score) * 100
            print(f"Artifiical Signal Cost lower than {100-percentile:.2f} of background pulsar cost distribution")

            #Create a range of x values for plotting the fitted normal curve
            xmin, xmax = min(data)-5, max(data)+5  # add some padding for visualization
            x_range = np.linspace(xmin, xmax, 300)
            pdf = norm.pdf(x_range, mu, sigma)

            if i == 0:
                # Plot the histogram and the fitted normal curve with labels for legend
                axs[0].hist(data, bins=20, density=True, alpha=0.6, color='skyblue',edgecolor='black', label="Background Pulsar Cost Distribution ")
                axs[0].plot(x_range, pdf, 'r', lw=2, label="Normal Distribution Curve of Pulsar Costs")
                axs[0].axvline(x=x_value, color='black', linestyle='--', lw=2, label="Artificial Signal Cost")
                axs[0].set_xlabel("Cost", fontsize = 22, fontweight = 'bold')
                axs[0].set_ylabel("Density", fontsize = 22, fontweight = 'bold')
                axs[0].tick_params(axis='both', labelsize=22)  # Adjust font size
                axs[0].set_title(f"ws = {wslist[0]}", fontsize = 22)
            else:
                # Plot the histogram and the fitted normal curve 
                axs[i].hist(data, bins=20, density=True, alpha=0.6, color='skyblue',edgecolor='black')
                axs[i].plot(x_range, pdf, 'r', lw=2)
                axs[i].axvline(x=x_value, color='black', linestyle='--', lw=2)
                axs[i].set_xlabel("Cost", fontsize = 22, fontweight = 'bold')
                axs[i].set_ylabel("Density", fontsize = 22, fontweight = 'bold')
                axs[i].tick_params(axis='both', labelsize=22)
                axs[i].set_title(f"ws = {wslist[i]}", fontsize = 22)

        

       
        # Get labels for legend and plot it
        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=20,bbox_to_anchor=(0.5, -0.0005))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leaves space for suptitle and legend
        plt.subplots_adjust(bottom=0.2)           # Pushes content up to make room
        plt.show()

    else:
        print("Invalid option. Please choose from: optimized pulsar, 2d star map, 3d star map, sequence figure, cost distribution graph.")