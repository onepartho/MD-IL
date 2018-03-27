import numpy as np
import pandas as pd
from scipy import stats
import scipy as sp
import os,csv
import statsmodels.api as sm
from scipy import interpolate
# === Matplotlib ===
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === Plot size ===
WIDTH = 5
rcParams['figure.figsize'] = WIDTH, WIDTH
rcParams['figure.dpi'] = 200

# === Color map ====
rcParams['image.cmap'] = 'gray'
rcParams['image.interpolation'] = 'none'

# === Axes Style ===
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['axes.labelweight'] = 'normal'
#rcParams['axes.facecolor'] = 'f2f2f2'
rcParams['axes.edgecolor'] = '444444' # GREY
rcParams['axes.axisbelow'] = True
rcParams['axes.labelcolor'] = '444444' # GREY

# === Font Style ===
rcParams['font.size'] = 11
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Stixsans']

#=== Latex ===
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}',r'\usepackage[notext]{stix}']

# === Legend Style ===
rcParams['legend.fontsize'] = 'medium'
rcParams['legend.frameon'] = True
rcParams['legend.numpoints'] = 3

# === Tick Style ===
rcParams['xtick.major.pad'] = 5
rcParams['xtick.major.width'] = 1
rcParams['xtick.color'] = '444444' # GREY

rcParams['ytick.major.pad'] = 5
rcParams['ytick.major.width'] = 1
rcParams['ytick.color'] = '444444' # GREY

rcParams['lines.linewidth'] = 2

# === Grid Style ===
rcParams['axes.grid'] = False
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = .25
rcParams['grid.color'] = '444444'
rcParams['grid.linewidth'] = 2


color_wheel = ['#332288', # blue
'#CC6677', # red
'#117733', # green
'#88CCEE', # light blue
'#44AA99',
'#999933',
'#882255',
'#DDCC77',
'#AA4499']
# == Colors ===
rcParams['axes.prop_cycle'] = plt.cycler('color', color_wheel)

pd.set_option('display.max_columns', 100)

##################################################################################################

def find_array_minimum_point(z_coord, array, z_min, z_max):
    from scipy import interpolate
    midpoint = 0.5*(z_min+z_max)
    array_itp = interpolate.interp1d(z_coord, array, kind='linear')
    fun = lambda closest_to_0 : abs(array_itp(closest_to_0)[0])
    bounds = [(z_min, z_max)]
    x = sp.optimize.minimize(fun, midpoint, bounds=bounds)
    #print x
    return x.x[0]


def lpot_rpot(z_coord, pot, z_mid1=4, z_mid2=6, z_end=10, field=None):
    """ If field is provided, defines midpoint as a place where field is closest to 0"""
    if np.any(field):
        smallest_field = find_array_minimum_point(field[:,0], field[:,1], z_mid1, z_mid2)
        mid_pot = np.mean(pot[z_coord <= smallest_field])
    else:
        mask = (z_coord > z_mid1) & (z_coord < z_mid2)
        mid_pot = np.mean(pot[mask])
    end_pot = np.mean(pot[z_coord > z_end])
    return mid_pot, end_pot - mid_pot

def num_derivC(pot, chg):
    """ Central difference numerical deriv using numpy."""
    dchg = np.gradient(chg)
    dpot = np.gradient(pot)
    return dchg/dpot

# Smoothing functions

from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    """http://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth(x,window_len=11,window='hanning'):
    """http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    """
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
