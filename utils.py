import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pyuvdata import UVCal, UVData, utils
import os
import sys
import glob
import uvtools as uvt
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle
import pandas
import warnings 
import copy
import utils
from hera_mc import cm_hookup
import math
from uvtools import dspec
import hera_qm 
warnings.filterwarnings('ignore')

def load_data(data_path,JD):
    HHfiles = sorted(glob.glob("{0}/zen.{1}.*.sum.uvh5".format(data_path,JD)))
    difffiles = sorted(glob.glob("{0}/zen.{1}.*.diff.uvh5".format(data_path,JD)))
    Nfiles = len(HHfiles)
    hhfile_bases = map(os.path.basename, HHfiles)
    hhdifffile_bases = map(os.path.basename, difffiles)

    # choose one for single-file plots
    file_index = np.min([len(HHfiles)-1, 20])
    hhfile1 = HHfiles[len(HHfiles)//2]
    difffile1 = difffiles[len(difffiles)//2]
    if len(HHfiles) != len(difffiles):
        print('############################################################')
        print('######### DIFFERENT NUMBER OF SUM AND DIFF FILES ###########')
        print('############################################################')
    # Load data
    uvd_hh = UVData()

    uvd_hh.read(hhfile1, skip_bad_files=True)
    uvd_xx1 = uvd_hh.select(polarizations = -5, inplace = False)
    uvd_xx1.ants = np.unique(np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array]))
    # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'


    uvd_hh = UVData()

    uvd_hh.read(hhfile1, skip_bad_files=True) 
    uvd_yy1 = uvd_hh.select(polarizations = -6, inplace = False)
    uvd_yy1.ants = np.unique(np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array]))

   
    return HHfiles, difffiles, uvd_xx1, uvd_yy1

def plot_autos(uvdx, uvdy):
    ants = uvdx.get_ants()
    freqs = (uvdx.freq_array[0])*10**(-6)
    times = uvdx.time_array
    lsts = uvdx.lst_array
    
    Nants = len(ants)
    Nside = int(np.ceil(np.sqrt(Nants)))
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime

    xlim = (np.min(freqs), np.max(freqs))
    ylim = (60, 90)

    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2, Nside*2), dpi=75)

    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if k < Nants:
                px, = ax.plot(freqs, 10*np.log10(np.abs(uvdx.get_data((ants[k], ants[k]))[t_index])), color='r', alpha=0.75, linewidth=1)
                py, = ax.plot(freqs, 10*np.log10(np.abs(uvdy.get_data((ants[k], ants[k]))[t_index])), color='b', alpha=0.75, linewidth=1)
            
                ax.grid(False, which='both')
                ax.set_title(str(ants[k]), fontsize=14)
            
                if k == 0:
                    ax.legend([px, py], ['East1', 'North1'])
                    #ax.legend([px, py, px2, py2, px3, py3], ['East1', 'North1', 'East2', 'North2', 'East3', 'North3'], fontsize=12)
            
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r'$10\cdot\log_{10}$ amplitude', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel('freq (MHz)', fontsize=10)
            k += 1
    fig.show()

    
def plot_wfs(uvd, pol):
    amps = np.abs(uvd.data_array[:, :, :, pol].reshape(uvd.Ntimes, uvd.Nants_data, uvd.Nfreqs, 1))
    
    ants = uvd.get_ants()
    freqs = (uvd.freq_array[0])*10**(-6)
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    
    Nants = len(ants)
    Nside = int(np.ceil(np.sqrt(Nants)))
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
    
    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2,Nside*2), dpi=75)
    if pol == 0:
        fig.suptitle("waterfalls from {0} -- {1} East Polarization".format(times[0], times[-1]), fontsize=14)
    else:
        fig.suptitle("waterfalls from {0} -- {1} North Polarization".format(times[0], times[-1]), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            if k < Nants:
                auto_bl = (ants[k], ants[k])
                im = ax.imshow(np.log10(np.abs(amps[:, k , :, 0])), aspect='auto', rasterized=True,
                           interpolation='nearest', vmin = 6.5, vmax = 8, 
                           extent=[freqs[0], freqs[-1], np.max(lsts), np.min(lsts)])
        
                ax.set_title(str(ants[k]), fontsize=10)
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                [t.set_rotation(25) for t in ax.get_xticklabels()]
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.set_xlabel('Frequency (MHz)', fontsize=10)
            k += 1
        
    cbar_ax=fig.add_axes([0.95,0.15,0.02,0.7])        
    fig.colorbar(im, cax=cbar_ax)
    fig.show()

def plot_closure(uvd, triad_length, pol):
    """Plot closure phase for an example triad.
    Parameters
    ----------
    files : list of strings
        List of data filenames
    triad_length : float {14., 29.}
        Length of the triangle segment length. Must be 14 or 29.
    pol : str {xx, yy}
        Polarization to plot.
    Returns
    -------
    None
    """


    if triad_length == 14.:
        triad_list = [[0, 11, 12], [0, 1, 12], [1, 12, 13], [1, 2, 13],
                      [2, 13, 14], [11, 23, 24], [11, 12, 24], [12, 24, 25],
                      [12, 13, 25], [13, 25, 26], [13, 14, 26], [14, 26, 27],
                      [23, 36, 37], [23, 24, 37], [24, 37, 38], [24, 25, 38],
                      [25, 38, 39], [25, 26, 39], [26, 39, 40], [26, 27, 40],
                      [27, 40, 41], [36, 37, 51], [37, 51, 52], [37, 38, 52],
                      [38, 52, 53], [38, 39, 53], [39, 53, 54], [39, 40, 54],
                      [40, 54, 55], [40, 41, 55], [51, 66, 67], [51, 52, 67],
                      [53, 54, 69], [54, 69, 70], [54, 55, 70], [55, 70, 71],
                      [65, 66, 82], [66, 82, 83], [66, 67, 83], [67, 83, 84],
                      [70, 71, 87], [120, 121, 140], [121, 140, 141], [121, 122, 141],
                      [122, 141, 142], [122, 123, 142], [123, 142, 143], [123, 124, 143]]
    else:
        triad_list = [[0, 23, 25], [0, 2, 25], [1, 24, 26], [2, 25, 27], [11, 36, 38],
                      [11, 13, 38], [12, 37, 39], [12, 14, 39], [13, 38, 40], [14, 39, 41],
                      [23, 25, 52], [24, 51, 53], [24, 26, 53], [25, 52, 54], [25, 27, 54],
                      [26, 53, 55], [36, 65, 67], [36, 38, 67], [38, 67, 69], [38, 40, 69],
                      [39, 41, 70], [40, 69, 71], [51, 82, 84], [51, 53, 84], [52, 83, 85],
                      [52, 54, 85], [54, 85, 87], [83, 85, 120], [85, 120, 122], [85, 87, 122],
                      [87, 122, 124]]


    # Look for a triad that exists in the data
    for triad in triad_list:
        bls = [[triad[0], triad[1]], [triad[1], triad[2]], [triad[2], triad[0]]]
        triad_in = True
        for bl in bls:
            inds = uvd.antpair2ind(bl[0], bl[1], ordered=False)
            if len(inds) == 0:
                triad_in = False
                break
        if triad_in:
            break

    if not triad_in:
        raise ValueError('Could not find triad in data.')

    closure_ph = np.angle(uvd.get_data(triad[0], triad[1], pol)
                          * uvd.get_data(triad[1], triad[2], pol)
                          * uvd.get_data(triad[2], triad[0], pol))
    plt.imshow(closure_ph, aspect='auto', rasterized=True,
                           interpolation='nearest', cmap = 'twilight')
    
def plotNodeAveragedSummary(uv,HHfiles,jd,pols=['xx','yy'],baseline_groups=[],removeBadAnts=False):
    """
    Plots a summary of baseline correlations throughout a night for each baseline group specified, separated into inter-node and intra-node baselines, for each polarization specified.
    
    Parameters
    ----------
    uv: UVData object
        UVData object containing any file from the desired night of observation.
    HHfiles: List
        A list of all files to be looked at for the desired night of observation.
    jd: String
        The JD of the night of observation
    pols: List
        A list containing the desired polarizations to look at. Options are any polarization strings accepted by pyuvdata. 
    baseline_groups: []
        A list containing the baseline types to look at, formatted as (length, N-S separation, label (str)).
    removeBadAnts: Bool
        Option to flag seemingly dead antennas and remove them from the per-baseline-group averaging. 
    
    Returns
    -------
    badAnts: List
        A list specifying the antennas flagged as dead or non-correlating.
    """
    if baseline_groups == []:
        baseline_groups = [(14,0,'14m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE'),(29,0,'29m E-W'),(29,22,'29m SW-NE'),
                       (44,0,'44m E-W'),(58.5,0,'58m E-W'),(73,0,'73m E-W'),(87.6,0,'88m E-W'),
                      (102.3,0,'102m E-W')]
    fig,axs = plt.subplots(len(pols),2,figsize=(16,16))
    maxLength = 0
    cmap = plt.get_cmap('Blues')
    nodeMedians,lsts,badAnts=get_correlation_baseline_evolutions(uv,HHfiles,jd,
                                                                bl_type=baseline_groups,removeBadAnts=removeBadAnts)
    for group in baseline_groups:
        if group[0] > maxLength:
            maxLength = group[0]
    for group in baseline_groups:
        length = group[0]
        data = nodeMedians[group[2]]
        colorInd = float(length/maxLength)
        if len(data['inter']['xx']) == 0:
            continue
        for i in range(len(pols)):
            pol = pols[i]
            axs[i][0].plot(lsts, data['inter'][pol], color=cmap(colorInd), label=group[2])
            axs[i][1].plot(lsts, data['intra'][pol], color=cmap(colorInd), label=group[2])
            axs[i][0].set_ylabel('Median Correlation Metric')
            axs[i][0].set_title('Internode, Polarization %s' % pol)
            axs[i][1].set_title('Intranode, Polarization %s' % pol)
    axs[1][1].legend()
    axs[1][0].set_xlabel('LST (hours)')
    axs[1][1].set_xlabel('LST (hours)')
    return badAnts
    
def plotVisibilitySpectra(file,jd,badAnts=[],pols=['xx','yy']):
    """
    Plots visibility amplitude spectra for a set of redundant baselines, labeled by inter vs. intranode baselines.
    
    Parameters
    ---------
    file: String
        File to calculate the spectra from
    jd: String
        JD of the night 'file' was observed on
    badAnts: List
        A list of antennas not to include in the plot
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    """
    
    fig, axs = plt.subplots(4,2,figsize=(12,16))
    plt.subplots_adjust(wspace=0.25)
    uv = UVData()
    uv.read_uvh5(file)
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    baseline_groups = get_baseline_groups(uv)
    freqs = uv.freq_array[0]/1000000
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    obstime_start = Time(uv.time_array[0],format='jd',location=loc)
    startTime = obstime_start.sidereal_time('mean').hour
    JD = int(obstime_start.jd)
    j = 0
    for orientation in baseline_groups:
        bls = baseline_groups[orientation]
        for p in range(len(pols)):
            inter=False
            intra=False
            pol = pols[p]
            for i in range(len(bls)):
                ants = uv.baseline_to_antnums(bls[i])
                ant1 = ants[0]
                ant2 = ants[1]
                key1 = 'HH%i:A' % (ant1)
                n1 = x[key1].get_part_from_type('node')['E<ground'][1:]
                key2 = 'HH%i:A' % (ant2)
                n2 = x[key2].get_part_from_type('node')['E<ground'][1:]
                dat = np.mean(np.abs(uv.get_data(ant1,ant2,pol)),0)
                auto1 = np.mean(np.abs(uv.get_data(ant1,ant1,pol)),0)
                auto2 = np.mean(np.abs(uv.get_data(ant2,ant2,pol)),0)
                norm = np.sqrt(np.multiply(auto1,auto2))
                dat = np.divide(dat,norm)
                if ant1 in badAnts or ant2 in badAnts:
                    continue
                if n1 == n2:
                    if intra is False:
                        axs[j][p].plot(freqs,dat,color='blue',label='intranode')
                        intra=True
                    else:
                        axs[j][p].plot(freqs,dat,color='blue')
                else:
                    if inter is False:
                        axs[j][p].plot(freqs,dat,color='red',label='internode')
                        inter=True
                    else:
                        axs[j][p].plot(freqs,dat,color='red')
                axs[j][p].set_yscale('log')
                axs[j][p].set_title('%s: %s pol' % (orientation,pols[p]))
                if j == 0:
                    axs[0][0].legend()
                    axs[3][p].set_xlabel('Frequency (MHz)')
        axs[j][0].set_ylabel('log(|Vij|)')
        axs[j][1].set_yticks([])
        j += 1
    fig.suptitle('Visibility spectra (JD: %i)' % (JD))
    fig.subplots_adjust(top=.94,wspace=0.05)
    
def plot_antenna_positions(uv, badAnts=[]):
    """
    Plots the positions of all antennas that have data, colored by node.
    
    Parameters
    ----------
    uv: UVData object
        Observation to extract antenna numbers and positions from
    badAnts: List
        A list of flagged or bad antennas. These will be outlined in black in the plot. 
    """
    
    plt.figure(figsize=(12,10))
    nodes, antDict, inclNodes = generate_nodeDict(uv)
    N = len(nodes)
    cmap = plt.get_cmap('tab20')
    n = 0
    for node in sorted(inclNodes):
        color = cmap(round(20/N*n))
        n += 1
        ants = nodes[node]['ants']
        for antNum in ants:
            width = 0
            idx = np.argwhere(uv.antenna_numbers == antNum)[0][0]
            antPos = uv.antenna_positions[idx]
            if antNum in badAnts:
                width=5
            if antNum == ants[0]:
                if antNum in badAnts:
                    plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,
                            markeredgecolor='black',markeredgewidth=width, markerfacecolor="None")
                    plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,label=str(node),
                            markeredgecolor='black',markeredgewidth=0)
                else:
                    plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,label=str(node),
                            markeredgecolor='black',markeredgewidth=width)
            else:
                plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,
                        markeredgecolor='black',markeredgewidth=width)
            plt.text(antPos[1]-1.5,antPos[2],str(antNum))
    plt.legend(title='Node Number',bbox_to_anchor=(1.15,0.9),markerscale=0.5,labelspacing=1.5)
    plt.title('Antenna Locations')
    
def plot_lst_coverage(uvd):
    """
    Plots the LST coverage for a particular night.
    
    Parameters
    ----------
    uvd: UVData Object
        Object containing a whole night of data, used to extract the time array.
    """
    lsts = uvd.lst_array*3.819719
    fig = plt.figure(figsize=(12,10))
    plt.hist(lsts, bins=int(len(np.unique(lsts))/20), alpha=0.7)
    plt.xlabel('LST (hours)')
    plt.ylabel('Count')
    plt.title('LST Coverage')
    
def calcEvenOddAmpMatrix(sm,df,pols=['xx','yy'],nodes='auto', badThresh=0.5):
    """
    Calculates a matrix of phase correlations between antennas, where each pixel is calculated as (even/abs(even)) * (conj(odd)/abs(odd)), and then averaged across time and frequency.
    
    Paramters:
    ---------
    sm: UVData Object
        Sum observation.
    df: UVData Object
        Diff observation. Must be the same time of observation as sm. 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    nodes: String or List
        Nodes to include in matrix. Default is 'auto', which generates a list of all nodes included in the provided data files. 
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas.
    
    Returns:
    -------
    data: Dict
        Dictionary containing calculated values, formatted as data[polarization][ant1,ant2]. 
    badAnts: List
        List of antennas that were flagged as bad based on badThresh.
    """
    if sm.time_array[0] != df.time_array[0]:
        print('FATAL ERROR: Sum and diff files are not from the same observation!')
        return None
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(sm)
    nants = len(sm.antenna_numbers)
    data = {}
    antnumsAll = sort_antennas(sm)
    badAnts = []
    for p in range(len(pols)):
        pol = pols[p]
        data[pol] = np.empty((nants,nants))
        for i in range(len(antnumsAll)):
            thisAnt = []
            for j in range(len(antnumsAll)):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                s = sm.get_data(ant1,ant2,pol)
                d = df.get_data(ant1,ant2,pol)
                even = (s + d)/2
                even = np.divide(even,np.abs(even))
                odd = (s - d)/2
                odd = np.divide(odd,np.abs(odd))
                product = np.multiply(even,np.conj(odd))
                data[pol][i,j] = np.abs(np.average(product))
                thisAnt.append(np.abs(np.average(product)))
            if np.nanmedian(thisAnt) < badThresh and antnumsAll[i] not in badAnts:
                badAnts.append(antnumsAll[i])
    return data, badAnts


def plotCorrMatrix(uv,data,pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    """
    Plots a matrix representing the phase correlation of each baseline.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation used for calculating the correlation metric
    data: Dict
        Dictionary containing the correlation metric for each baseline and each polarization. Formatted as data[polarization]  [ant1,ant2] 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    vminIn: float
        Lower limit of colorbar. Default is 0.
    vmaxIn: float
        Upper limit of colorbar. Default is 1.
    nodes: Dict
        Dictionary containing the nodes (and their constituent antennas) to include in the matrix. Formatted as nodes[Node #][Ant List, Snap # List, Snap Location List].
    logScale: Bool
        Option to put colormap on a logarithmic scale. Default is False.
    """
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.antenna_numbers)
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,len(pols),figsize=(16,16))
    dirs = ['NS','EW']
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    antnumsAll = sort_antennas(uv)
    for p in range(len(pols)):
        pol = pols[p]
        nants = len(antnumsAll)
        if logScale is True:
            im = axs[p].imshow(data[pol],cmap='plasma',origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
        else:
            im = axs[p].imshow(data[pol],cmap='plasma',origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
        axs[p].set_xticks(np.arange(0,nantsTotal)+1)
        axs[p].set_xticklabels(antnumsAll,rotation=90)
        axs[p].xaxis.set_ticks_position('top')
        axs[p].set_title('polarization: ' + dirs[p] + '\n')
        n=0
        for node in sorted(inclNodes):
            n += len(nodeDict[node]['ants'])
            axs[p].axhline(len(antnumsAll)-n+.5,lw=4)
            axs[p].axvline(n+.5,lw=4)
            axs[p].text(n-len(nodeDict[node]['ants'])/2,-.5,node)
        axs[p].text(.42,-.07,'Node Number',transform=axs[p].transAxes)
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[1].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
    axs[1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[1].transAxes)
    axs[1].set_yticklabels([])
    axs[1].set_yticks([])
    axs[0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0].set_yticklabels(antnumsAll)
    axs[0].set_ylabel('Antenna Number')
    cbar_ax = fig.add_axes([0.95,0.53,0.02,0.38])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Correlation Matrix - JD: %s, LST: %.0fh' % (str(jd),np.round(lst,0)))
    fig.subplots_adjust(top=1.32,wspace=0.05)
    
def get_hourly_files(uv, HHfiles, jd):
    """
    Generates a list of files spaced one hour apart throughout a night of observation, and the times those files were observed.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the given night, used only for grabbing the telescope location
    HHFiles: List
        List of all files from the desired night of observation
    jd: String
        JD of the night of observation
        
    Returns:
    -------
    use_files: List
        List of files separated by one hour
    use_lsts: List
        List of LSTs of observations in use_files
    """
    use_lsts = []
    use_files = []
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    for file in HHfiles:
        try:
            dat = UVData()
            dat.read(file, read_data=False)
        except KeyError:
            continue
        jd = dat.time_array[0]
        t = Time(jd,format='jd',location=loc)
        lst = round(t.sidereal_time('mean').hour,2)
        if np.abs((lst-np.round(lst,0)))<0.05:
            if len(use_lsts)>0 and np.abs(use_lsts[-1]-lst)<0.5:
                if np.abs((lst-np.round(lst,0))) < abs((use_lsts[-1]-np.round(lst,0))):
                    use_lsts[-1] = lst
                    use_files[-1] = file
            else:
                use_lsts.append(lst)
                use_files.append(file)
    return use_files, use_lsts

def get_baseline_groups(uv, bl_groups=[(14,0,'14m E-W'),(29,0,'29m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE')]):
    """
    Generate dictionary containing baseline groups.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation to extract antenna position information from
    bl_groups: List
        Desired baseline types to extract, formatted as (length (float), N-S separation (float), label (string))
        
    Returns:
    --------
    bls: Dict
        Dictionary containing list of lists of redundant baseline numbers, formatted as bls[group label]
    """
    
    bls={}
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        for group in bl_groups:
            if np.abs(lengths[i]-group[0])<1:
                ant1 = uv.baseline_to_antnums(bl[0])[0]
                ant2 = uv.baseline_to_antnums(bl[0])[1]
                antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                disp = (antPos2-antPos1)[0][0]
                if np.abs(disp[2]-group[1])<0.5:
                    bls[group[2]] = bl
    return bls


    
def get_correlation_baseline_evolutions(uv,HHfiles,jd,badThresh=0.35,pols=['xx','yy'],bl_type=(14,0,'14m E-W'),
                                        removeBadAnts=False, plotMatrix=True):
    """
    Calculates the average correlation metric for a set of redundant baseline groups at one hour intervals throughout a night of observation.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the desired night, used only for getting telescope location information.
    HHfiles: List
        List of all files for a night of observation
    jd: String
        JD of the given night of observation
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas. Default is 0.35.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    bl_type: Tuple
        Redundant baseline group to calculate correlation metric for. Default is 14m E-W baselines
    removeBadAnts: Bool
        Option to exclude antennas marked as bad from calculation. Default is False.
    plotMatrix: Bool
        Option to plot the correlation matrix for observations once each hour. Default is True.
        
    Returns:
    -------
    result: Dict
        Per hour correlation metric, formatted as result[baseline type]['inter' or 'intra'][polarization]
    lsts: List
        LSTs that metric was calculated for, spaced 1 hour apart.
    bad_antennas: List
        Antenna numbers flagged as bad based on badThresh parameter.
    """
    files, lsts = get_hourly_files(uv, HHfiles, jd)
    nTimes = len(files)
    plotTimes = [0,nTimes-1,nTimes//2]
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    JD = math.floor(uv.time_array[0])
    bad_antennas = []
    corrSummary = generateDataTable(uv)
    result = {}
    for f in range(nTimes):
        file = files[f]
        sm = UVData()
        df = UVData()
        try:
            sm.read(file, skip_bad_files=True)
            df.read('%sdiff%s' % (file[0:-8],file[-5:]), skip_bad_files=True)
        except:
            continue
        matrix, badAnts = calcEvenOddAmpMatrix(sm,df,nodes='auto',badThresh=badThresh)
        if plotMatrix is True and f in plotTimes:
            plotCorrMatrix(sm, matrix, nodes='auto')
        for group in bl_type:
            medians = {
                'inter' : {},
                'intra' : {}
                }
            for pol in pols:
                medians['inter'][pol] = []
                medians['intra'][pol] = []
            if file == files[0]:
                result[group[2]] = {
                    'inter' : {},
                    'intra' : {}
                }
                for pol in pols:
                    result[group[2]]['inter'][pol] = []
                    result[group[2]]['intra'][pol] = []
            bls = get_baseline_type(uv,bl_type=group)
            if bls == None:
                continue
            baselines = [uv.baseline_to_antnums(bl) for bl in bls]
            for ant in badAnts:
                if ant not in bad_antennas:
                    bad_antennas.append(ant)
            if removeBadAnts is True:
                nodeInfo = {
                    'inter' : getInternodeMedians(sm,matrix,badAnts=bad_antennas, baselines=baselines),
                    'intra' : getIntranodeMedians(sm,matrix,badAnts=bad_antennas, baselines=baselines)
                }
            else:
                nodeInfo = {
                    'inter' : getInternodeMedians(sm,matrix, baselines=baselines),
                    'intra' : getIntranodeMedians(sm,matrix,baselines=baselines)
                }
            for node in nodeDict:
                for pol in pols:
                    corrSummary[node][pol]['inter'].append(nodeInfo['inter'][node][pol])
                    corrSummary[node][pol]['intra'].append(nodeInfo['intra'][node][pol])
                    medians['inter'][pol].append(nodeInfo['inter'][node][pol])
                    medians['intra'][pol].append(nodeInfo['intra'][node][pol])
            for pol in pols:
                result[group[2]]['inter'][pol].append(np.nanmedian(medians['inter'][pol]))
                result[group[2]]['intra'][pol].append(np.nanmedian(medians['intra'][pol]))
    return result,lsts,bad_antennas

def generateDataTable(uv,pols=['xx','yy']):
    """
    Simple helper function to generate an empty dictionary of the format desired for get_correlation_baseline_evolutions()
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
        
    Returns:
    -------
    dataObject: Dict
        Empty dict formatted as dataObject[node #][polarization]['inter' or 'intra']
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    dataObject = {}
    for node in nodeDict:
        dataObject[node] = {}
        for pol in pols:
            dataObject[node][pol] = {
                'inter' : [],
                'intra' : []
            }
    return dataObject

def getInternodeMedians(uv,data,pols=['xx','yy'],badAnts=[],baselines='all'):
    """
    Identifies internode baseliens and performs averaging of correlation metric.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    data: Dict
        Dictionary containing correlation metric information, formatted as data[polarization][ant1,ant2].
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
    badAnts: List
        List of antennas that have been flagged as bad - if provided, they will be excluded from averaging.
    baselines: List
        List of baseline types to include in calculation.
        
    Returns:
    -------
    nodeMeans: Dict
        Per-node averaged correlation metrics, formatted as nodeMeans[node #][polarization].
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    antnumsAll=sort_antennas(uv)
    nants = len(antnumsAll)
    nodeMeans = {}
    nodeCorrs = {}
    for node in nodeDict:
        nodeCorrs[node] = {}
        nodeMeans[node] = {}
        for pol in pols:
            nodeCorrs[node][pol] = []        
    start=0
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    for pol in pols:
        for i in range(nants):
            for j in range(nants):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                if ant1 not in badAnts and ant2 not in badAnts and ant1 != ant2:
                    if baselines=='all' or (ant1,ant2) in baselines:
                        key1 = 'HH%i:A' % (ant1)
                        n1 = x[key1].get_part_from_type('node')['E<ground'][1:]
                        key2 = 'HH%i:A' % (ant2)
                        n2 = x[key2].get_part_from_type('node')['E<ground'][1:]
                        dat = data[pol][i,j]
                        if n1 != n2:
                            nodeCorrs[n1][pol].append(dat)
                            nodeCorrs[n2][pol].append(dat)
    for node in nodeDict:
        for pol in pols:
            nodeMeans[node][pol] = np.nanmedian(nodeCorrs[node][pol])
    return nodeMeans

def getIntranodeMedians(uv, data, pols=['xx','yy'],badAnts=[],baselines='all'):
    """
    Identifies intranode baseliens and performs averaging of correlation metric.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    data: Dict
        Dictionary containing correlation metric information, formatted as data[polarization][ant1,ant2].
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
    badAnts: List
        List of antennas that have been flagged as bad - if provided, they will be excluded from averaging.
    baselines: List
        List of baseline types to include in calculation.
        
    Returns:
    -------
    nodeMeans: Dict
        Per-node averaged correlation metrics, formatted as nodeMeans[node #][polarization].
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    antnumsAll=sort_antennas(uv)
    nodeMeans = {}
    start=0
    for node in nodeDict:
        nodeMeans[node]={}
        for pol in pols:
            nodeCorrs = []
            for i in range(start,start+len(nodeDict[node]['ants'])):
                for j in range(start,start+len(nodeDict[node]['ants'])):
                    ant1 = antnumsAll[i]
                    ant2 = antnumsAll[j]
                    if ant1 not in badAnts and ant2 not in badAnts and i != j:
                        if baselines=='all' or (ant1,ant2) in baselines:
                            nodeCorrs.append(data[pol][i,j])
            nodeMeans[node][pol] = np.nanmedian(nodeCorrs)
        start += len(nodeDict[node]['ants'])
    return nodeMeans

def get_baseline_type(uv,bl_type=(14,0,'14m E-W')):
    """
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to get baseline information from.
    bl_type: Tuple
        Redundant baseline group to extract baseline numbers for. Formatted as (length, N-S separation, label).
    
    Returns:
    -------
    bl: List
        List of lists of redundant baseline numbers. Returns None if the provided bl_type is not found.
    """
    
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        if np.abs(lengths[i]-bl_type[0])<1:
            ant1 = uv.baseline_to_antnums(bl[0])[0]
            ant2 = uv.baseline_to_antnums(bl[0])[1]
            antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
            antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
            disp = (antPos2-antPos1)[0][0]
            if np.abs(disp[2]-bl_type[1])<0.5:
                return bl
    return None

def generate_nodeDict(uv):
    """
    Generates dictionaries containing node and antenna information.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    
    Returns:
    -------
    nodes: Dict
        Dictionary containing entry for all nodes, each of which has keys: 'ants', 'snapLocs', 'snapInput'.
    antDict: Dict
        Dictionary containing entry for all antennas, each of which has keys: 'node', 'snapLocs', 'snapInput'.
    inclNodes: List
        Nodes that have hooked up antennas.
    """
    
    antnums = uv.antenna_numbers
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    nodes = {}
    antDict = {}
    inclNodes = []
    for ant in antnums:
        key = 'HH%i:A' % (ant)
        n = x[key].get_part_from_type('node')['E<ground'][1:]
        snapLoc = (x[key].hookup['E<ground'][-1].downstream_input_port[-1], ant)
        snapInput = (x[key].hookup['E<ground'][-2].downstream_input_port[1:], ant)
        antDict[ant] = {}
        antDict[ant]['node'] = str(n)
        antDict[ant]['snapLocs'] = snapLoc
        antDict[ant]['snapInput'] = snapInput
        inclNodes.append(n)
        if n in nodes:
            nodes[n]['ants'].append(ant)
            nodes[n]['snapLocs'].append(snapLoc)
            nodes[n]['snapInput'].append(snapInput)
        else:
            nodes[n] = {}
            nodes[n]['ants'] = [ant]
            nodes[n]['snapLocs'] = [snapLoc]
            nodes[n]['snapInput'] = [snapInput]
    inclNodes = np.unique(inclNodes)
    return nodes, antDict, inclNodes

def sort_antennas(uv):
    """
    Helper function that sorts antennas by snap input number.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation used for extracting node and antenna information.
        
    Returns:
    -------
    sortedAntennas: List
        All antennas with data, sorted into order of ascending node number, and within that by ascending snap number, and within that by ascending snap input number.
    """
    
    nodes, antDict, inclNodes = generate_nodeDict(uv)
    sortedAntennas = []
    for n in sorted(inclNodes):
        snappairs = []
        h = cm_hookup.Hookup()
        x = h.get_hookup('HH')
        for ant in nodes[n]['ants']:
            snappairs.append(antDict[ant]['snapLocs'])
        snapLocs = {}
        locs = []
        for pair in snappairs:
            ant = pair[1]
            loc = pair[0]
            locs.append(loc)
            if loc in snapLocs:
                snapLocs[loc].append(ant)
            else:
                snapLocs[loc] = [ant]
        locs = sorted(np.unique(locs))
        ants_sorted = []
        for loc in locs:
            ants = snapLocs[loc]
            inputpairs = []
            for ant in ants:
                key = 'HH%i:A' % (ant)
                pair = (int(x[key].hookup['E<ground'][-2].downstream_input_port[1:]), ant)
                inputpairs.append(pair)
            for _,a in sorted(inputpairs):
                ants_sorted.append(a)
        for ant in ants_sorted:
            sortedAntennas.append(ant)
    return sortedAntennas

def clean_ds(HHfiles, difffiles, bls, area=1000., tol=1e-9, skip_wgts=0.2): 
    
    uvd_ds = UVData()
    uvd_ds.read(HHfiles[0], bls=bls[0], polarizations=-5, keep_all_metadata=False)
    times = np.unique(uvd_ds.time_array)
    Nfiles = int(1./((times[-1]-times[0])*24.))
    try:
        uvd_ds.read(HHfiles[:Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
        uvd_ds.flag_array = np.zeros_like(uvd_ds.flag_array)
        uvd_diff = UVData()
        uvd_diff.read(difffiles[:Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
    except:
        uvd_ds.read(HHfiles[len(HHfiles)//2:len(HHfiles)//2+Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
        uvd_ds.flag_array = np.zeros_like(uvd_ds.flag_array)
        uvd_diff = UVData()
        uvd_diff.read(difffiles[len(HHfiles)//2:len(HHfiles)//2+Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
    
    uvf_m, uvf_fws = hera_qm.xrfi.xrfi_h1c_pipe(uvd_ds, sig_adj=1, sig_init=3)
    hera_qm.xrfi.flag_apply(uvf_m, uvd_ds)
    
    freqs = uvd_ds.freq_array[0]
    FM_idx = np.searchsorted(freqs*1e-6, [85,110])
    flag_FM = np.zeros(freqs.size, dtype=bool)
    flag_FM[FM_idx[0]:FM_idx[1]] = True
    win = dspec.gen_window('bh7', freqs.size)
    
    pols = ['nn','ee']
    
    _data_sq_cleaned, data_rs = {}, {}
    for bl in bls:
        for i, pol in enumerate(pols):
            key = (bl[0],bl[1],pol)
            data = uvd_ds.get_data(key)
            diff = uvd_diff.get_data(key)
            wgts = (~uvd_ds.get_flags(key)*~flag_FM[np.newaxis,:]).astype(float)

            d_even = (data+diff)*0.5
            d_odd = (data-diff)*0.5
            d_even_cl, d_even_rs, _ = dspec.high_pass_fourier_filter(d_even, wgts, area*1e-9, freqs[1]-freqs[0], 
                                                                     tol=tol, skip_wgt=skip_wgts, window='bh7')
            d_odd_cl, d_odd_rs, _ = dspec.high_pass_fourier_filter(d_odd, wgts, area*1e-9, freqs[1]-freqs[0],
                                                                   tol=tol, skip_wgt=skip_wgts, window='bh7')

            idx = np.where(np.mean(np.abs(d_even_cl), axis=1) == 0)[0]
            d_even_cl[idx] = np.nan
            d_even_rs[idx] = np.nan        
            idx = np.where(np.mean(np.abs(d_odd_cl), axis=1) == 0)[0]
            d_odd_cl[idx] = np.nan
            d_odd_rs[idx] = np.nan

            _d_even = np.fft.fftshift(np.fft.ifft((d_even_cl+d_even_rs)*win), axes=1)
            _d_odd = np.fft.fftshift(np.fft.ifft((d_odd_cl+d_odd_rs)*win), axes=1)

            _data_sq_cleaned[key] = _d_odd.conj()*_d_even
            data_rs[key] = d_even_rs
        
    return _data_sq_cleaned, data_rs, uvd_ds, uvd_diff

def plot_wfds(uvd, _data_sq, pol):    
    ants = uvd.get_ants()
    freqs = uvd.freq_array[0]
    times = uvd.time_array
    lsts = uvd.lst_array
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
    Nants = len(ants)
    Nside = int(np.ceil(np.sqrt(Nants)))
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
    
    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2,Nside*2), dpi=75)
    if pol == 'ee':
        fig.suptitle("delay spectrum (auto) waterfalls from {0} -- {1} East Polarization".format(times[0], times[-1]), fontsize=14)
    else:
        fig.suptitle("delay spectrum (auto) waterfalls from {0} -- {1} North Polarization".format(times[0], times[-1]), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            if k < Nants:
                key = (ants[k], ants[k], pol)
                ds = 10.*np.log10(np.sqrt(np.abs(_data_sq[key])/np.abs(_data_sq[key]).max(axis=1)[:,np.newaxis]))
                im = ax.imshow(ds, aspect='auto', rasterized=True,
                           interpolation='nearest', vmin = -50, vmax = -30, 
                           extent=[taus[0]*1e9, taus[-1]*1e9, np.max(lsts), np.min(lsts)])
        
                ax.set_title(str(ants[k]), fontsize=10)
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                [t.set_rotation(25) for t in ax.get_xticklabels()]
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.set_xlabel('Delay (ns)', fontsize=10)
            k += 1
        
    cbar_ax=fig.add_axes([0.95,0.15,0.02,0.7])        
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('dB')
    fig.show()
    
def plot_ds(uvd, uvd_diff, _data_sq_cleaned, data_rs, skip_wgts=0.2):            
    matplotlib.rcParams.update({'font.size': 8})
    freqs = uvd.freq_array[0]
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    FM_idx = np.searchsorted(freqs*1e-6, [85,110])
    flag_FM = np.zeros(freqs.size, dtype=bool)
    flag_FM[FM_idx[0]:FM_idx[1]] = True
    
    ants = np.sort(uvd.get_ants())
    pols = ['nn','ee']
    colors = ['b','r']
    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd)

    for i, ant in enumerate(ants):
        i2 = i % 2
        if(i2 == 0):
            fig = plt.figure(figsize=(10, 3), dpi=110)
            grid = plt.GridSpec(5, 4, wspace=0.55, hspace=2)
        for j, pol in enumerate(pols):
            key = (ant,ant,pol)

            ax = fig.add_subplot(grid[:5, 0+i2*2])

            ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
            _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
            ns_ave = np.abs(np.nanmean(_diff, axis=0))
            ns_ave_ = np.sqrt(np.abs(np.mean(_diff.conj()*_diff, axis=0))/(2*_diff.shape[0]))
            norm = np.max(ds_ave)
            plt.plot(taus*1e9, 10*np.log10(ds_ave/norm), color=colors[j], label=pols[j], linewidth=0.7)
            plt.plot(taus*1e9, 10*np.log10(ns_ave/norm), color=colors[j], alpha=0.5, linewidth=0.5)
            plt.plot(taus*1e9, 10*np.log10(ns_ave_/norm), color=colors[j], ls='--', linewidth=0.5)
            plt.axvspan(250, 500, alpha=0.2, facecolor='y')
            plt.axvspan(2500, 3000, alpha=0.2, facecolor='g')
            plt.axvspan(3500, 4000, alpha=0.2, facecolor='m')
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=False)
            plt.xlim(0,4500)
            plt.ylim(-60,0)
            plt.title('ant {}'.format(ant))
            plt.xlabel(r'$\tau$ (ns)')
            plt.ylabel(r'$|\tilde{V}(\tau)|$ in dB')
            for yl in range(-50,0,10):
                plt.axhline(y=yl, color='k', lw=0.5, alpha=0.1)
            if(j == 0):
                plt.annotate('node {} (snap {})'.format(antDict[ant]['node'],antDict[ant]['snapLocs'][0]), xy=(0.04,0.930), xycoords='axes fraction')

            ax2 = fig.add_subplot(grid[:3, 1+i2*2])
            auto = np.abs(uvd.get_data(key))
            auto_ave = np.mean(auto/np.median(auto, axis=1)[:,np.newaxis], axis=0)
            wgts = (~uvd.get_flags(key)*~flag_FM[np.newaxis,:]).astype(float)
            wgts = np.where(wgts > skip_wgts, wgts, 0)
            auto_flagged = auto/wgts
            auto_flagged[np.isinf(auto_flagged)] = np.nan
            auto_flagged_ave = np.nanmean(auto_flagged/np.nanmedian(auto_flagged, axis=1)[:,np.newaxis], axis=0)

            plt.plot(freqs/1e6, np.log10(auto_ave), linewidth=1.0, color=colors[j], label='autocorrelation')
            plt.plot(freqs/1e6, np.log10(auto_flagged_ave*0.7), linewidth=1.0, color=colors[j], alpha=0.5)
            if(j == 0):
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False, handlelength=0, handletextpad=0)
            plt.xlim(freqs.min()/1e6, freqs.max()/1e6)
            plt.ylabel(r'log$_{10}(|V(\nu)|)$')
            plt.ylim(-1, 0.7)

            ax3 = fig.add_subplot(grid[3:5, 1+i2*2])
            data_rs_ave = np.nanmean(data_rs[key]/np.nanmedian(auto_flagged, axis=1)[:,np.newaxis], axis=0)
            plt.plot(freqs/1e6, np.log10(np.abs(data_rs_ave)), linewidth=1.0, color=colors[j], label='clean residual')
            if(j == 0):
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), frameon=False, handlelength=0, handletextpad=0)
            plt.xlim(freqs.min()/1e6, freqs.max()/1e6)
            plt.ylim(-5,0)
            plt.xlabel(r'$\nu$ (MHz)')
            plt.ylabel(r'log$_{10}(|V(\nu)|)$')
    matplotlib.rcParams.update({'font.size': 10})
            
def plot_ds_diagnosis(uvd_diff, _data_sq_cleaned):
    freqs = uvd_diff.freq_array[0]
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
    ants = np.sort(uvd_diff.get_ants())
    pols = ['nn','ee']
    colors = ['b','r']
    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd_diff)
    marker = ['o', 'x', 'P', 's', 'd', 'p', '<', 'h', '+', '>', 'X', '*']
    
    nodes = []
    ds_ave_250_500_nn, ds_ave_250_500_ee = [], []
    ds_peak_2500_3000_nn, ds_peak_2500_3000_ee = [], []
    ds_ratio_3500_4000_nn, ds_ratio_3500_4000_ee = [], []
    for ant in ants:
        nodes.append(int(antDict[ant]['node']))
        for pol in pols:
            key = (ant,ant,pol)            
            ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
            _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
            ns_ave = np.sqrt(np.abs(np.mean(_diff.conj()*_diff, axis=0))/(2*_diff.shape[0]))
            norm = np.max(ds_ave)
            idx_region1 = (np.abs(taus)*1e9 > 250) * (np.abs(taus)*1e9 < 500)
            idx_region2 = (np.abs(taus)*1e9 > 2500) * (np.abs(taus)*1e9 < 3000)
            idx_region_out = (np.abs(taus)*1e9 > 3000) * (np.abs(taus)*1e9 < 3200)
            idx_region3 = (np.abs(taus)*1e9 > 3500) * (np.abs(taus)*1e9 < 4000)
            if(pol == 'nn'):
                ds_ave_250_500_nn.append(np.nanmean(ds_ave[idx_region1]/norm))
                ds_peak_2500_3000_nn.append(np.std(ds_ave[idx_region2])/np.std(ds_ave[idx_region_out]))
                ds_ratio_3500_4000_nn.append(np.nanmean(ds_ave[idx_region3])/np.mean(ns_ave[idx_region3]))
            else:
                ds_ave_250_500_ee.append(np.nanmean(ds_ave[idx_region1]/norm))
                ds_peak_2500_3000_ee.append(np.std(ds_ave[idx_region2])/np.std(ds_ave[idx_region_out]))
                ds_ratio_3500_4000_ee.append(np.nanmean(ds_ave[idx_region3])/np.mean(ns_ave[idx_region3]))
    nodes = np.array(nodes)
    N_nodes = nodes.size
    ds_ave_250_500_nn = np.array(ds_ave_250_500_nn)
    ds_ave_250_500_ee = np.array(ds_ave_250_500_ee)
    ds_peak_2500_3000_nn = np.array(ds_peak_2500_3000_nn)
    ds_peak_2500_3000_ee = np.array(ds_peak_2500_3000_ee)
    ds_ratio_3500_4000_nn = np.array(ds_ratio_3500_4000_nn)
    ds_ratio_3500_4000_ee = np.array(ds_ratio_3500_4000_ee)
            
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    node_number = np.unique(nodes)
    for i, node in enumerate(node_number):
        idx_ant = np.where(nodes == node)[0]
        plt.plot(ants[idx_ant], 10*np.log10(ds_ave_250_500_nn[idx_ant]), 'bo', label='nn', marker=marker[i%N_nodes])
        plt.plot(ants[idx_ant], 10*np.log10(ds_ave_250_500_ee[idx_ant]), 'ro', label='ee', marker=marker[i%N_nodes])
        if(i == 0):
            plt.legend()
    for i, ant in enumerate(ants):
        plt.annotate('{}'.format(ant), xy=(ants[i], 10*np.log10(ds_ave_250_500_nn[i])))
    plt.xlabel('Antenna number')
    plt.ylabel('dB (relative to the peak)')
    plt.title('Averaged delay spectrum at 250-500 ns')
    plt.subplot(3,1,2)
    for i, node in enumerate(node_number):
        idx_ant = np.where(nodes == node)[0]
        plt.plot(ants[idx_ant], ds_peak_2500_3000_nn[idx_ant], 'bo', label='nn', marker=marker[i%N_nodes])
        plt.plot(ants[idx_ant], ds_peak_2500_3000_ee[idx_ant], 'ro', label='ee', marker=marker[i%N_nodes])
        if(i == 0):
            plt.legend()
    for i, ant in enumerate(ants):
        plt.annotate('{}'.format(ant), xy=(ants[i], ds_peak_2500_3000_nn[i]))
    plt.axhline(y=1, ls='--', color='k')
    plt.xlabel('Antenna number')
    plt.ylabel('$\sigma_{2500-3000}/\sigma_{3000-3200}$')
    plt.title('Standard deviation ratio between 2500-3000 ns and 3000-3200 ns')
    plt.subplot(3,1,3)
    for i, node in enumerate(node_number):
        idx_ant = np.where(nodes == node)[0]
        plt.plot(ants[idx_ant], 10*np.log10(ds_ratio_3500_4000_nn[idx_ant]), 'bo', label='nn', marker=marker[i%N_nodes])
        plt.plot(ants[idx_ant], 10*np.log10(ds_ratio_3500_4000_ee[idx_ant]), 'ro', label='ee', marker=marker[i%N_nodes])
        if(i == 0):
            plt.legend()
    for i, ant in enumerate(ants):
        plt.annotate('{}'.format(ant), xy=(ants[i], 10*np.log10(ds_ratio_3500_4000_nn[i])))
    plt.axhline(y=0, ls='--', color='k')
    plt.xlabel('Antenna number')
    plt.ylabel('Deviation from the noise level in dB')
    plt.title('Deviation of averaged delay spectrum at 3500-4000 ns from the noise level')
    plt.subplots_adjust(hspace=0.4)
    
def plot_ds_nodes(uvd_diff, _data_sq_cleaned):
    freqs = uvd_diff.freq_array[0]
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
    ants = np.sort(uvd_diff.get_ants())
    pols = ['nn','ee']
    colors = ['b','r']
    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd_diff)
    
    nodes = []
    ds_ave_nn, ns_ave_nn = [], []
    ds_ave_ee, ns_ave_ee = [], []
    for ant in ants:
        nodes.append(int(antDict[ant]['node']))
        for pol in pols:
            key = (ant,ant,pol)
            ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
            _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
            ns_ave = np.abs(np.nanmean(_diff, axis=0))
            norm = np.max(ds_ave)
            if(pol == 'nn'):
                ds_ave_nn.append(ds_ave/norm)
                ns_ave_nn.append(ns_ave/norm)
            else:
                ds_ave_ee.append(ds_ave/norm)
                ns_ave_ee.append(ns_ave/norm)
                
    nodes = np.array(nodes)
    ds_ave_nn = np.array(ds_ave_nn).reshape(ants.size,taus.size)
    ns_ave_nn = np.array(ns_ave_nn).reshape(ants.size,taus.size)
    ds_ave_ee = np.array(ds_ave_ee).reshape(ants.size,taus.size)
    ns_ave_ee = np.array(ns_ave_ee).reshape(ants.size,taus.size)
    node_number = np.unique(nodes)
    for i, node in enumerate(node_number):
        if(i % 2 == 0):
            plt.figure(figsize=(16,4))
        for j, pol in enumerate(pols):
            plt.subplot(1,4,2*(i%2)+j+1)
            idx_ant = np.where(nodes == node)[0]
            if(pol == 'nn'):
                for idx in idx_ant:
                    plt.plot(taus*1e9, 10*np.log10(ds_ave_nn[idx]), label='({},{})'.format(ants[idx],ants[idx]), linewidth=0.8)
            else:                
                for idx in idx_ant:
                    plt.plot(taus*1e9, 10*np.log10(ds_ave_ee[idx]), label='({},{})'.format(ants[idx],ants[idx]), linewidth=0.8)
            plt.xlim(0,4500)
            plt.ylim(-55,0)
            plt.title('node {}, {}'.format(node, pol))
            plt.xlabel(r'$\tau$ (ns)')
            if(i % 2 == 0 and j == 0):
                plt.ylabel(r'$|\tilde{V}(\tau)|$ in dB')
            plt.grid(axis='y')        
            plt.legend(loc='upper left', ncol=2)
        
def plot_wfds_cr(uvd, _data_sq, pol):
    bls = uvd.get_antpairs()
    freqs = uvd.freq_array[0]
    times = uvd.time_array
    lsts = uvd.lst_array
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    idx_mid = np.where(taus == 0)[0]
    
    Nants = len(uvd.get_antpairs())
    Nside = int(np.ceil(np.sqrt(Nants)))
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
    
    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2,Nside*2), dpi=75)
    if pol == 'ee':
        fig.suptitle("delay spectrum (cross) waterfalls from {0} -- {1} East Polarization".format(times[0], times[-1]), fontsize=14)
    else:
        fig.suptitle("delay spectrum (cross) waterfalls from {0} -- {1} North Polarization".format(times[0], times[-1]), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            if k < Nants:
                key = (bls[k][0], bls[k][1], pol)
                ds = 10.*np.log10(np.sqrt(np.abs(_data_sq[key])/np.abs(_data_sq[key][:,idx_mid])))
                im = ax.imshow(ds, aspect='auto', rasterized=True,
                           interpolation='nearest', vmin = -30, vmax = 0, 
                           extent=[taus[0]*1e9, taus[-1]*1e9, np.max(lsts), np.min(lsts)])
        
                ax.set_title(str(bls[k]), fontsize=10)
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                [t.set_rotation(25) for t in ax.get_xticklabels()]
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.set_xlabel('Delay (ns)', fontsize=10)
            k += 1
        
    cbar_ax=fig.add_axes([0.95,0.15,0.02,0.7])        
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('dB')
    fig.show()
def plot_metric(metrics, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
    '''Helper function for quickly plotting an individual antenna metric.'''

    if ants is None:
        ants = list(set([key[0] for key in metrics.keys()]))
    if antpols is None:
        antpols = list(set([key[1] for key in metrics.keys()]))
    for antpol in antpols:
        for i,ant in enumerate(ants):
            metric = 0
            if (ant,antpol) in metrics:
                metric = metrics[(ant,antpol)]
            plt.plot(i,metric,'.')
            plt.annotate(str(ant)+antpol,xy=(i,metrics[(ant,antpol)]))
        plt.gca().set_prop_cycle(None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
def show_metric(ant_metrics, antmetfiles, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
    print("Ant Metrics for {}".format(antmetfiles[1]))
    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['meanVij'],
            title = 'Mean Vij Modified z-Score')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorr'],
            title = 'Redundant Visibility Correlation Modified z-Score')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['meanVijXPol'], antpols=['n'],
            title = 'Modified z-score of (Vxy+Vyx)/(Vxx+Vyy)')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['n'],
            title = 'Modified z-Score of Power Correlation Ratio Cross/Same')
    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['e'],
            title = 'Modified z-Score of Power Correlation Ratio Cross/Same')

def all_ant_mets(antmetfiles,HHfiles):
    file = HHfiles[0]
    uvd_hh = UVData()
    uvd_hh.read_uvh5(file)
    uvdx = uvd_hh.select(polarizations = -5, inplace = False)
    uvdx.ants = np.unique(np.concatenate([uvdx.ant_1_array, uvdx.ant_2_array]))
    ants = uvdx.get_ants()
    times = uvd_hh.time_array
    Nants = len(ants)
    jd_start = np.floor(times.min())
    antfinfiles = []
    for i,file in enumerate(antmetfiles):
        if i%50==0:
            antfinfiles.append(antmetfiles[i])
    Nfiles = len(antfinfiles)
    Nfiles2 = len(antmetfiles)
    xants = np.zeros((Nants*2, Nfiles2))
    dead_ants = np.zeros((Nants*2, Nfiles2))
    cross_ants = np.zeros((Nants*2, Nfiles2))
    badants = []
    pol2ind = {'n':0, 'e':1}
    times = []

    for i,file in enumerate(antfinfiles):
        time = file[54:60]
        times.append(time)

    for i,file in enumerate(antmetfiles):
        antmets = hera_qm.ant_metrics.load_antenna_metrics(file)
        for j in antmets['xants']:
            xants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
        badants.extend(map(lambda x: x[0], antmets['xants']))
        for j in antmets['crossed_ants']:
            cross_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
        for j in antmets['dead_ants']:
            dead_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1

    badants = np.unique(badants)
    xants[np.where(xants==1)] *= np.nan
    dead_ants[np.where(dead_ants==0)] *= np.nan
    cross_ants[np.where(cross_ants==0)] *= np.nan

    antslabels = []
    for i in ants:
        labeln = str(i) + 'n'
        labele = str(i) + 'e'
        antslabels.append(labeln)
        antslabels.append(labele)

    fig, ax = plt.subplots(1, figsize=(16,20), dpi=75)

    # plotting
    ax.matshow(xants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])
    ax.matshow(dead_ants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])
    ax.matshow(cross_ants, aspect='auto', cmap='RdBu', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])

    # axes
    ax.grid(color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(times))+0.5)
    ax.set_yticks(np.arange(Nants*2)+0.5)
    ax.tick_params(size=8)

    if Nfiles > 20:
        ticklabels = times
        ax.set_xticklabels(ticklabels)
    else:
        ax.set_xticklabels(times)

    ax.set_yticklabels(antslabels)

    [t.set_rotation(30) for t in ax.get_xticklabels()]
    [t.set_size(12) for t in ax.get_xticklabels()]
    [t.set_rotation(0) for t in ax.get_yticklabels()]
    [t.set_size(12) for t in ax.get_yticklabels()]

    ax.set_title("Ant Metrics bad ants over observation", fontsize=14)
    ax.set_xlabel('decimal of JD = {}'.format(int(jd_start)), fontsize=16)
    ax.set_ylabel('antenna number and pol', fontsize=16)
    red_ptch = mpatches.Patch(color='red')
    grn_ptch = mpatches.Patch(color='green')
    blu_ptch = mpatches.Patch(color='blue')
    ax.legend([red_ptch, blu_ptch, grn_ptch], ['dead ant', 'cross ant', 'good ant'], fontsize=14)
