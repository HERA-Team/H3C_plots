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
warnings.filterwarnings('ignore')

def load_data(data_path):
    HHfiles = sorted(glob.glob("{0}/zen.*.*.HH.uvh5".format(data_path)))
    difffiles = sorted(glob.glob("{0}/zen.*.*.HH.diff.uvh5".format(data_path)))
    Nfiles = len(HHfiles)
    hhfile_bases = map(os.path.basename, HHfiles)
    hhdifffile_bases = map(os.path.basename, difffiles)

    # choose one for single-file plots
    file_index = np.min([len(HHfiles)-1, 20])
    hhfile1 = HHfiles[len(HHfiles)//2]
    difffile1 = difffiles[len(difffiles)//2]
    if len(HHfiles) != len(difffiles):
        print('############################################################')
        print('############### SUM AND DIFF FILE MISMATCH #################')
        print('############################################################')
    # Load data
    uvd_hh = UVData()
    uvd_diff = UVData()
    uvd_sum = UVData()
    
    uvd_diff.read_uvh5(difffile1)
    uvd_sum.read(hhfile1)

    uvd_hh.read_uvh5(hhfile1)
    uvd_xx1 = uvd_hh.select(polarizations = -5, inplace = False)
    uvd_xx1.ants = np.unique(np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array]))
    # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'


    uvd_hh = UVData()

    uvd_hh.read_uvh5(hhfile1) 
    uvd_yy1 = uvd_hh.select(polarizations = -6, inplace = False)
    uvd_yy1.ants = np.unique(np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array]))

    #first file 
    uvdfirst = UVData()
    uvdfirst.read_uvh5(HHfiles[0:1], polarizations=[-5, -6])

    #last file
    uvdlast = UVData()
    uvdlast.read_uvh5(HHfiles[-1], polarizations=[-5, -6])
   
    return HHfiles, uvd_xx1, uvd_yy1, uvdfirst, uvdlast, uvd_diff, uvd_sum


def plot_autos(uvdx, uvdy, uvd1, uvd2):
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
    lsts = uvd.lst_array
    
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


def calcEvenOddAmpMatrix(sm,df,pols=['xx','yy'],nodes='auto',freq='avg',metric='amplitude',flagging=False):
    if sm.time_array[0] != df.time_array[0]:
        print('FATAL ERROR: Sum and diff files are not from the same observation!')
        return None
    if nodes=='auto':
        nodeDict = generate_nodeDict(sm)
    nants = len(sm.antenna_numbers)
    data = {}
    antnumsAll = []
    for node in nodeDict:
        snapLocs = []
        nodeAnts = []
        for ant in nodeDict[node]['ants']:
            nodeAnts.append(ant)
        for snapLoc in nodeDict[node]['snapLocs']:
            snapLocs.append(snapLoc)
        snapSorted = [x for _,x in sorted(zip(snapLocs,nodeAnts))]
        for ant in snapSorted:
            antnumsAll.append(ant)
    for p in range(len(pols)):
        pol = pols[p]
        data[pol] = np.empty((nants,nants))
        for i in range(len(antnumsAll)):
            for j in range(len(antnumsAll)):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                if freq=='avg':
                    s = sm.get_data(ant1,ant2,pol)
                    d = df.get_data(ant1,ant2,pol)
                    sflg = np.invert(sm.get_flags(ant1,ant2,pol))
                    dflg = np.invert(df.get_flags(ant1,ant2,pol))
                elif len(freq)==1:
                    freqs = sm.freq_array[0]
                    freqind = (np.abs(freqs - freq*1000000)).argmin()
                    s = sm.get_data(ant1,ant2,pol)[:,freqind]
                    d = df.get_data(ant1,ant2,pol)[:,freqind]
                    sflg = np.invert(sm.get_flags(ant1,ant2,pol)[:,freqind])
                    dflg = np.invert(df.get_flags(ant1,ant2,pol)[:,freqind])
                else:
                    freqs = sm.freq_array[0]
                    freqindHigh = (np.abs(freqs - freq[-1]*1000000)).argmin()
                    freqindLow = (np.abs(freqs - freq[0]*1000000)).argmin()
                    s = sm.get_data(ant1,ant2,pol)[:,freqindLow:freqindHigh]
                    d = df.get_data(ant1,ant2,pol)[:,freqindLow:freqindHigh]
                    sflg = np.invert(sm.get_flags(ant1,ant2,pol)[:,freqindLow:freqindHigh])
                    dflg = np.invert(df.get_flags(ant1,ant2,pol)[:,freqindLow:freqindHigh])
                if flagging is True:
                    flags = np.logical_and(sflg,dflg)
                    s = s[flags]
                    d = d[flags]
                even = (s + d)/2
                even = np.divide(even,np.abs(even))
                odd = (s - d)/2
                odd = np.divide(odd,np.abs(odd))
                product = np.multiply(even,np.conj(odd))
                if metric=='amplitude':
                    data[pol][i,j] = np.abs(np.average(product))
                elif metric=='phase':
                    product = np.average(product)
                    re = np.real(product)
                    imag = np.imag(product)
                    phase = np.arctan(np.divide(imag,re))
                    data[pol][i,j] = phase
                else:
                    print('Invalid metric')
    return data


def plotCorrMatrix(uv,data,freq='All',pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    if nodes=='auto':
        nodeDict = generate_nodeDict(uv)
    nantsTotal = len(uv.antenna_numbers)
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,len(pols),figsize=(16,16))
    dirs = ['NS','EW']
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    t = Time(uv.time_array[0],format='jd',location=loc)
    t.format='fits'
    jd = int(uv.time_array[0])
    antnumsAll = []
    for node in nodeDict:
        snapLocs = []
        nodeAnts = []
        for ant in nodeDict[node]['ants']:
            nodeAnts.append(ant)
        for snapLoc in nodeDict[node]['snapLocs']:
            snapLocs.append(snapLoc)
        snapSorted = [x for _,x in sorted(zip(snapLocs,nodeAnts))]
        for ant in snapSorted:
            antnumsAll.append(ant)
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
        for node in nodeDict:
            n += len(nodeDict[node]['ants'])
            axs[p].axhline(len(antnumsAll)-n+.5,lw=4)
            axs[p].axvline(n+.5,lw=4)
            axs[p].text(n-len(nodeDict[node]['ants'])/2,-.4,node)
        axs[p].text(.42,-.07,'Node Number',transform=axs[p].transAxes)
    n=0
    for node in nodeDict:
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
    #fig.suptitle('JD: ' + str(jd) + ', Frequency Range: ' + '%i-%iMHz' % (freq[0],freq[1]))
    fig.suptitle(str(jd) + ' Even*conj(Odd) Normalized Visibility Amplitude')
    fig.subplots_adjust(top=1.32,wspace=0.05)
    
def generate_nodeDict(uv):
    antnums = uv.antenna_numbers
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    nodes = {}
    for ant in antnums:
        key = 'HH%i:A' % (ant)
        n = x[key].get_part_in_hookup_from_type('node')['E<ground'][2]
        snapLoc = x[key].hookup['E<ground'][-1].downstream_input_port[-1]
        if n in nodes:
            nodes[n]['ants'].append(ant)
            nodes[n]['snapLocs'].append(snapLoc)
        else:
            nodes[n] = {}
            nodes[n]['ants'] = [ant]
            nodes[n]['snapLocs'] = [snapLoc]
    return nodes


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
    
def plot_antenna_positions(uv):
    plt.figure(figsize=(12,10))
    nodes = generate_nodeDict(uv)
    N = len(nodes)
    colors = ['b','g','y','r','c','m']
    n = 0
    for node in nodes:
        color = colors[n]
        n += 1
        ants = nodes[node]['ants']
        for antNum in ants:
            idx = np.argwhere(uv.antenna_numbers == antNum)[0][0]
            antPos = uv.antenna_positions[idx]
            if antNum == ants[0]:
                plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,label=str(node))
            else:
                plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5)
            plt.text(antPos[1]-1.5,antPos[2],str(antNum))
    plt.legend(title='Node Number',bbox_to_anchor=(1.15,0.9),markerscale=0.5,labelspacing=1.5)
    plt.title('Antenna Locations')
            
