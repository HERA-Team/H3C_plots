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
warnings.filterwarnings('ignore')

def load_data(data_path):
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
   
    return HHfiles, difffiles, uvd_xx1, uvd_yy1, uvdfirst, uvdlast
    

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
    
def plotNodeAveragedSummary(uv,HHfiles,pols=['xx','yy'],baseline_groups=[],removeBadAnts=False):
    baseline_groups = [(14,0,'14m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE'),(29,0,'29m E-W'),(29,22,'29m SW-NE'),
                   (44,0,'44m E-W'),(58.5,0,'58m E-W'),(73,0,'73m E-W'),(87.6,0,'88m E-W'),
                  (102.3,0,'102m E-W')]
    fig,axs = plt.subplots(len(pols),2,figsize=(16,16))
    maxLength = 0
    cmap = plt.get_cmap('Blues')
    nodeMedians,lsts,badAnts=get_correlation_baseline_evolutions(uv,HHfiles,bl_type=baseline_groups,removeBadAnts=removeBadAnts)
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
    
def plotVisibilitySpectra(file,badAnts=[],length=29,pols=['xx','yy'], clipLowAnts=True):
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
                n1 = x[key1].get_part_from_type('node')['E<ground'][2]
                key2 = 'HH%i:A' % (ant2)
                n2 = x[key2].get_part_from_type('node')['E<ground'][2]
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
    plt.figure(figsize=(12,10))
    nodes, antDict, inclNodes = generate_nodeDict(uv)
    N = len(nodes)
    cmap = plt.get_cmap('tab20')
    n = 0
    labelled = []
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
                plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,label=str(node),
                        markeredgecolor='black',markeredgewidth=width)
                labelled.append(node)
            else:
                plt.plot(antPos[1],antPos[2],marker="h",markersize=40,color=color,alpha=0.5,
                        markeredgecolor='black',markeredgewidth=width)
            plt.text(antPos[1]-1.5,antPos[2],str(antNum))
    plt.legend(title='Node Number',bbox_to_anchor=(1.15,0.9),markerscale=0.5,labelspacing=1.5)
    plt.title('Antenna Locations')
    
def calcEvenOddAmpMatrix(sm,df,pols=['xx','yy'],nodes='auto', badThresh=0.5):
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


def plotCorrMatrix(uv,data,freq='All',pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.antenna_numbers)
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,len(pols),figsize=(16,16))
    dirs = ['NS','EW']
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    t = Time(uv.time_array[0],format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    jd = uv.time_array[0]
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
            axs[p].text(n-len(nodeDict[node]['ants'])/2,-.4,node)
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
    
def get_hourly_files(uv, HHfiles):
    use_lsts = []
    use_files = []
    for file in HHfiles:
        filename = file.split('zen.',1)[1]
        jd = float(filename[0:-5])
        loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
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


    
def get_correlation_baseline_evolutions(uv,HHfiles,badThresh=0.35,pols=['xx','yy'],bl_type=(14,0,'14m E-W'),
                                        removeBadAnts=False, plotMatrix=True):
    files, lsts = get_hourly_files(uv, HHfiles)
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
        sm.read_uvh5(file)
        df.read_uvh5('%s.diff%s' % (file[0:-5],file[-5:]))
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
                        n1 = x[key1].get_part_from_type('node')['E<ground'][2]
                        key2 = 'HH%i:A' % (ant2)
                        n2 = x[key2].get_part_from_type('node')['E<ground'][2]
                        dat = data[pol][i,j]
                        if n1 != n2:
                            nodeCorrs[n1][pol].append(dat)
                            nodeCorrs[n2][pol].append(dat)
    for node in nodeDict:
        for pol in pols:
            nodeMeans[node][pol] = np.nanmedian(nodeCorrs[node][pol])
    return nodeMeans

def getIntranodeMedians(uv, data, pols=['xx','yy'],badAnts=[],baselines='all'):
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
    antnums = uv.antenna_numbers
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    nodes = {}
    antDict = {}
    inclNodes = []
    for ant in antnums:
        key = 'HH%i:A' % (ant)
        n = x[key].get_part_from_type('node')['E<ground'][2]
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
