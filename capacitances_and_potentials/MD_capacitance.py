import numpy as np
import pandas as pd
from scipy import stats
import scipy as sp
import os,csv
import statsmodels.api as sm
from scipy import interpolate
import MD_potential
from MD_potential import *

# === Matplotlib ===
import matplotlib.pyplot as plt
from matplotlib import rcParams

def conv_smooth(x, y, window_size=11, window='hanning', extension=100):
    """ Convolve signal with hanning window"""
    # extend function to makes sure edges behave nicely
    dx = x[1] - x[0]
    x_itp_long = np.arange(x.min() - extension, x.max() + extension, dx)
    y_itp_long_f = sp.interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    y_itp_long = y_itp_long_f(x_itp_long)
    # after convolution part of the function close to the edges behave poorly
    # so we cut them
    l_cut = (window_size - 1)/2
    r_cut = window_size - 1 - l_cut
    # convolution magic from
    # http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    s = np.r_[y_itp_long[window_size-1:0:-1], y_itp_long, y_itp_long[-1:-window_size:-1]]
    w = eval('np.' + window + '(window_size)')
    y_smooth = np.convolve(w/w.sum(), s, mode='valid')[l_cut:-r_cut]
    # return to non-extended function
    return y_smooth[(x_itp_long > x[0]-dx/2.0) & (x_itp_long < x[-1]+dx/2.0)]


def num_derivC(pot, chg):
    """ Central difference numerical deriv using numpy."""
    dchg = np.gradient(chg)
    dpot = np.gradient(pot)
    return dchg/dpot

def plot_window(ax, x, window_size, window, plot_size=.1):
    w = eval('np.' + window + '(window_size)')
    dx = x[1]-x[0]
    x_plot = np.arange(0, window_size*dx - dx*.1, dx)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    func = w*(y_max - y_min)*plot_size + y_min
    ax.plot(x_plot + x_max - x_plot[-1], func, 'k-')
    
def compute_cap(gromacs_data, itp_points=1000, w='hamming', fine_ws=61, coarse_ws=201, chg_left=-20,
                chg_right=20):
    # interpolate to increase point size
    # and regularize grid
    pot, chg = gromacs_data[:,0], gromacs_data[:, 1]
    chg_itp = np.linspace(chg.min(),chg.max(), itp_points)
    pot_itp_f = interpolate.interp1d(chg, pot, kind='linear')
    pot_itp = pot_itp_f(chg_itp)

    # Convolution - smoothing
    # convolve with hamming window
    # optimally, we want to describe low surface charges with less (fine) smoothing (since we have a lot of simulations there)
    # and smooth pot-chg relation for higher charges to larger extent (coarse smoothing)

    # transition speeeds from coarse to fine and from fine to coarse smoothings
    trans_left = 5.0
    trans_right = 5.0

    # smoothed pots
    fine_pot = conv_smooth(chg_itp, pot_itp, fine_ws, window=w)
    coarse_pot = conv_smooth(chg_itp, pot_itp, coarse_ws, window=w)

    # merge two potentials at two different points
    sigma1 =  1. / (1 + np.exp(-(chg_itp - chg_left) / trans_left));
    sigma2 =  1. / (1 + np.exp(-(chg_itp - chg_right) / trans_right));
    merged = (1-sigma1) * coarse_pot + sigma1 * fine_pot
    merged = (1-sigma2) * merged + sigma2 * coarse_pot
    
    cap = num_derivC(merged, chg_itp)
    return np.c_[chg_itp, merged, cap]

def plot_cap(gromacs_data, rel_pos=0, itp_points=1000, w='hamming', fine_ws=151, coarse_ws=201, chg_left=-20,
             chg_right=20):
    chg_pot_cap = compute_cap(gromacs_data, itp_points=itp_points, w=w, fine_ws=fine_ws,
                              coarse_ws=coarse_ws, chg_left=chg_left, chg_right=chg_right)
    pot, chg = gromacs_data[:,0], gromacs_data[:,1]
    chg_itp, conv_pot, cap = chg_pot_cap[:,0], chg_pot_cap[:,1], chg_pot_cap[:,2]
    #plt.plot(pot_itp, chg_itp , label='itp', alpha=.5)
    fine_ws=61; coarse_ws=20
    fig, (ax1,ax2)  = plt.subplots(ncols=2, figsize=(9, 4))
    ax1.plot(conv_pot, chg_itp, '-', label='conv')#, lw=3)
    ax1.plot(pot, chg, '.', c='red', label='data')
    ax1.set_xlabel('U (V)')
    ax1.set_ylabel(r'$\sigma$ $(\mathrm{mC\cdot cm^{-2}})$')
    ax1.legend(loc='best')
    ax1.set_xlim(pot.min(), pot.max())
    ax1.set_ylim(chg.min(), chg.max())
    ax2.plot(chg_pot_cap[:,1],chg_pot_cap[:,2],  label='fit')
    ax2.legend(loc='best')
    ax2.set_xlabel('U (V)')
    ax2.set_ylabel(r'$C$ $(\mathrm{\mu F\cdot cm^{-2}})$')
    #plt.savefig('../gfx/u_pot.png',dpi=300)

    print 'Capacitance maximum: {}'.format(np.amax(chg_pot_cap[:,2]))
    print 'Corresponding charge: {}'.format(chg_pot_cap[:,0][np.argmax(chg_pot_cap[:,2])])
    print 'Corresponding potential: {}'.format(chg_pot_cap[:,1][np.argmax(chg_pot_cap[:,2])])
    return fig, (ax1, ax2)

#plot_cap(compute_potentials(simulations, rel_pos=0.102),w='hamming',fine_ws=261, coarse_ws=200, chg_left=-20, chg_right=20)
