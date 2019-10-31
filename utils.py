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
    

def make_wfs(files, antennas, uvd_old, poli=0):
    _oldd = {}
    _d = {}
    for ant in antennas:
        auto_bl = (ant, ant)
        _oldd[auto_bl] = uvd_old.get_data(auto_bl)   
    for i, fl in enumerate(files):
        uvd_new = UVData()
        uvd_new.read_uvh5(fl, polarizations=[-5, -6])
        antennas = uvd_new.get_ants()
        for ant in antennas: 
            auto_bl = (ant, ant)
            if i==0:
                d_old = _oldd[auto_bl]
            else:
                d_old = _d[auto_bl] 
            d_new = uvd_new.get_data(auto_bl)
            _d[auto_bl] = np.concatenate((d_old, d_new), axis=0) 

    return _d 

    
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


def calcEvenOddAmpMatrix(sm,df,pols=['xx','yy'],nodes='auto',freq='avg',metric='amplitude'):
    if not freq == 'avg':
        freqs = uv.freq_array[0]
        freqind = (np.abs(freqs - freq*1000000)).argmin()
    if nodes=='auto':
        nodes = generate_nodeDict(sm)
    nants = len(sm.antenna_numbers)
    data = {}
    antnumsAll = []
    for node in nodes:
        for ant in nodes[node]:
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
                else:
                    s = sm.get_data(ant1,ant2,pol)[:,freqind]
                    d = df.get_data(ant1,ant2,pol)[:,freqind]
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


def plotCorrMatrix(uv,data,pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    jd = uv.time_array[0]
    if nodes=='auto':
        nodes = generate_nodeDict(uv)
    nantsTotal = len(uv.antenna_numbers)
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,len(pols),figsize=(16,16))
    dirs = ['NS','EW']
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    t = Time(uv.time_array[0],format='jd',location=loc)
    t.format='fits'
    antnumsAll = []
    for node in nodes:
        for ant in nodes[node]:
            antnumsAll.append(ant)
    for p in range(len(pols)):
        pol = pols[p]
        nants = len(antnumsAll)
        if logScale is True:
            im = axs[p].imshow(
                data[pol],cmap='plasma',origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],
                norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
        else:
            im = axs[p].imshow(
                data[pol],cmap='plasma',origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],
                vmin=vminIn, vmax=vmaxIn)
        axs[p].set_xticks(np.arange(0,nantsTotal)+1)
        axs[p].set_xticklabels(antnumsAll,rotation=90)
        axs[p].xaxis.set_ticks_position('top')
        axs[p].set_title('polarization: ' + dirs[p] + '\n')
        n=0
        for node in nodes:
            n += len(nodes[node])
            axs[p].axhline(len(antnumsAll)-n+.5,lw=4)
            axs[p].axvline(n+.5,lw=4)
            axs[p].text(n-len(nodes[node])/2,-.4,node)
        axs[p].text(.42,-.07,'Node Number',transform=axs[p].transAxes)
    n=0
    for node in nodes:
        n += len(nodes[node])
        axs[1].text(nantsTotal+1,nantsTotal-n+len(nodes[node])/2,node)
    axs[1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[1].transAxes)
    axs[1].set_yticklabels([])
    axs[1].set_yticks([])
    axs[0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0].set_yticklabels(antnumsAll)
    axs[0].set_ylabel('Antenna Number')
    cbar_ax = fig.add_axes([0.95,0.53,0.02,0.38])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
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
        if n in nodes:
            nodes[n].append(ant)
        else:
            nodes[n] = [ant]
    return nodes