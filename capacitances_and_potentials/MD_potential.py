import numpy as np
from scipy import stats
import scipy as sp
import pandas as pd
import os,csv
import statsmodels.api as sm
from scipy import interpolate

# === Matplotlib ===
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === Logging ===
import logging
logging.basicConfig()
log=logging.getLogger('ipynb')
log.setLevel(logging.DEBUG)
log.info('logging initialized')

##############################################

def find_array_minimum_point(z_coord, array, z_min, z_max):
    """ 
    Find the smallest absolute value of array within given range, 
    """
    from scipy import interpolate
    midpoint = 0.5*(z_min+z_max)
    # create interpolation function for array
    array_itp = interpolate.interp1d(z_coord, array, kind='linear')
    # return absolute value of the interpolated function
    abs_array_itp = lambda z_coord : abs(array_itp(z_coord)[0])
    bounds = [(z_min, z_max)]
    # minimize absolute value of the array, subject to bounds
    x = sp.optimize.minimize(abs_array_itp, midpoint, bounds=bounds)
    return x.x[0]

def lpot_rpot(z_coord, pot, z_mid1=4, z_mid2=6, field=None, z_end=None, debug=False):
    """ Return potential drops on left and right electrodes, given coordinates and potential.
    If field is provided, defines midpoint as a place where field is closest to 0 within z_mid1, z_mid2 range.
    
    In debug mode in addition to two potentials, returns the location of minimum field.
    """
    if not z_end:
        z_end = z_coord[-1]
    if np.any(field):
        smallest_field = find_array_minimum_point(field[:,0], field[:,1], z_mid1, z_mid2)
        # evaluate potential in the region +/- 2*dz from the smallest field location
        dz = z_coord[1] - z_coord[0]
        mask = (z_coord > smallest_field - 2*dz) & (z_coord < smallest_field + 2*dz)
        mid_pot = np.mean(pot[mask])
    else:
        mask = (z_coord > z_mid1) & (z_coord < z_mid2)
        mid_pot = np.mean(pot[mask])
    end_pot = np.mean(pot[z_coord > z_end])
    if debug:
        return mid_pot, end_pot - mid_pot, smallest_field
    else:
        return mid_pot, end_pot - mid_pot

class ChgSimulation(object):
    """A utility object used to describe a series of simulations at a certain surface charge."""
    eps0 = 5.5268e-2
    chg_eps = 1.0e-6 # level for non-zero charge
    
    def __init__(self, chg_sim_dir, chg_name='charge.xvg', replicas=None, field_name=None, 
                 pot_name=None, log=None, walls=None, set_chg=False, eps_rf=1):
        """
        chg_sim_dir : string
            directory in which simulations with a constant surface charge are located. The final part of
            the path must include surface charge value for simulations. For example: my/long/path/3.0
            
        chg_name : string, default charge.xvg
            the name of the files containing charge distribution in simulation box
        
        replicas : list-like, default None
            names of directories in chg_sim_dir (without full_path), containing individual simulations.
            If None is provided, will analyze all directories in chg_sim_dir.
            
        field_name : string, default None
            the name of the files containing field distribution in simulation box. If none, will be
            computed from the charge distribution
            
        pot_name : string, default None
            the name of the files containing potential distribution in simulation box. If none, will be
            computed from the charge distribution
            
        walls : array or string, default None
            location of left and right wall (electrodes). If none - will try to guess it using charge distribution
            If walls=='edges', assumes that walls are located at the edges of provided z-coordinates
        
        set_chg : Float, default False
            Set surface charge to walls
            
        eps_rf : float, default 1
            Dielectric constant throught the box. Used to downscale walls
            
        log : object, default None
            an object of python logging module
        """
        # init stuff
        if log:
            self.log = log
        else:
            self.log = logging.getLogger(name='charge simulation logger')
            self.log.setLevel(logging.INFO)
        self.chg_sim_dir = chg_sim_dir
        self.chg_name = chg_name
        if not replicas:
            replicas = [d for d in os.path.listdir(chg_sim_dir) if os.path.isdir(os.path.join(chg_sim_dir, d))]
        self.replicas = [os.path.join(chg_sim_dir, replica) for replica in replicas]
        self.field_name = field_name
        self.pot_name = pot_name
        if chg_sim_dir.endswith('/'):
            chg_sim_dir = chg_sim_dir[:-1]
        self.surf_chg = float(os.path.split(chg_sim_dir)[1])
        self.eps_rf = eps_rf
        self.no_field_points = []  # locations near the center of the box where field is close to 0
        self.chgs   = []  # charge distributions
        self._get_chgs()
        self.dz = self.chgs[0][1,0] - self.chgs[0][0,0] # step between grid points
        if walls=='edges':
            # leave a bit of space
            self.lelec = [self.dz]*len(replicas)
            self.relec = [self.chgs[i][-2,0] for i in range(len(replicas))]
        elif walls:
            self.lelec = [walls[0]]*len(replicas)
            self.relec = [walls[1]]*len(replicas)
        else:
            self.lelec = []  # locations of left electrode
            self.relec = []  # locations of left electrode
            self._find_walls()
        if set_chg:
            self._set_wall_charge()
        self.fields = []  # field distributions
        self._get_fields()
        self._find_midpoints()
        self.inbetween_fields = [] # the field strengths in the regions between electrode and IL
        #self._compute_inbetween_fields()


    def _get_chgs(self):
        """Load charges from replicas"""
        for replica in self.replicas:
            self.log.debug('Finding walls for replica: {}'.format(replica))
            chg = np.loadtxt(os.path.join(replica, self.chg_name), comments=['#','@'])
            self.chgs.append(chg)
                
    def _find_walls(self):
        """find positions of left and right elctrode in each replica using the locations of the 
        first points, where charge is non-zero. To deal with chg=0 we add an offset of 10*dz
        """
        for i, chg in enumerate(self.chgs):
            self.log.debug('Finding walls for replica: {}'.format(self.replicas[i]))
            dz = chg[1,0]-chg[0,0]
            non_zero_chg = np.abs(chg[:,1]) > self.chg_eps
            self.lelec.append(chg[:,0][non_zero_chg][0])
            self.relec.append(chg[:,0][non_zero_chg][-1])
            if abs(self.surf_chg) < 1.0e-3:
                self.lelec[-1] = self.lelec[-1] - 10*dz
                self.relec[-1] = self.relec[-1] + 10*dz
            self.log.debug('Left wall found at: {}'.format(self.lelec[-1]))
            self.log.debug('Right wall found at: {}'.format(self.relec[-1]))
            
    def _set_wall_charge(self):
        microC_per_cm2_to_electron_per_nm2_conv = 0.062415091
        chg_density = self.surf_chg*microC_per_cm2_to_electron_per_nm2_conv/self.dz
        for i, (lelec, relec) in enumerate(zip(self.lelec, self.relec)):
            left_wall_mask = np.isclose(self.chgs[i][:,0], lelec, atol=self.dz/2)
            right_wall_mask = np.isclose(self.chgs[i][:,0], relec, atol=self.dz/2)
            assert sum(left_wall_mask) == 1
            assert sum(right_wall_mask) == 1
            self.chgs[i][:,1][left_wall_mask] =   chg_density/self.eps_rf
            self.chgs[i][:,1][right_wall_mask] = -chg_density/self.eps_rf

    def _get_fields(self):
        """load/computed field distributions in the boxes"""
        if self.field_name:
            for replica in self.replicas:
                self.fields.append(np.loadtxt(os.path.join(replica, self.field_name), comments=['#','@']))
        else:
            # compute field ourselves
            for chg in self.chgs:
                field_f = interpolate.InterpolatedUnivariateSpline(chg[:,0], chg[:,1], k=1).antiderivative()
                self.fields.append(np.c_[chg[:,0], field_f(chg[:,0])/ChgSimulation.eps0/self.eps_rf])
        
    def _find_midpoints(self):
        """find positions within +/- 1 [in provided length units] from the middle of the box, 
        where the field is smallest
        """
        for i, replica in enumerate(self.replicas):
            mid = 0.5*(self.lelec[i] + self.relec[i])
            self.log.debug("mid point: {}".format(mid))
            # create interpolation function for array
            array_itp = interpolate.interp1d(self.fields[i][:,0], self.fields[i][:,1], kind='linear')
            # return absolute value of the interpolated function
            abs_array_itp = lambda z_coord : abs(array_itp(z_coord)[0])
            bounds = [(mid-1, mid+1)]  # +/- 1 from mid in sim units
            self.log.debug("wall bounds {}".format(bounds))
            # minimize absolute value of the array, subject to bounds
            x = sp.optimize.minimize(abs_array_itp, mid, bounds=bounds)
            self.no_field_points.append(x.x[0])
            
    def _compute_inbetween_fields(self):
        """find the strength of the field between electrode and IL.
        Assumes it is equal to the strength of the field immediately (10*dz) after the location of electrode wall.
        Does this for both left and right electrode and throws error if they are not very similar for
        any of the replicas
        """
        for i, field in enumerate(self.fields):
            lfield = np.mean(field[:,1][np.isclose(field[:,0], self.lelec[i] + 2*self.dz, atol=self.dz)])
            rfield = np.mean(field[:,1][np.isclose(field[:,0], self.relec[i] - 2*self.dz, atol=self.dz)])
            
            if not np.isclose(lfield, rfield, atol=.1, rtol=1.0e-2):
                plt.plot(field[:,0], field[:,1])
                plt.plot(self.chgs[i][:,0], self.chgs[i][:,1])
                plt.hlines(lfield, self.lelec[i], self.relec[i], 'red', linestyles='--')
                plt.hlines(rfield, self.lelec[i], self.relec[i], 'green', linestyles='--')
                plt.xlim(self.lelec[i] - 15*self.dz, self.lelec[i] + 15*self.dz)
                plt.vlines([self.lelec[i] + 3*self.dz,  self.relec[i] - 3*self.dz], field[:,1].min(), field[:,1].max())
                self.log.debug('dz: {}'.format(self.dz))
                plt.show()
                plt.plot(field[:,0], field[:,1])
                plt.plot(self.chgs[i][:,0], self.chgs[i][:,1])
                plt.hlines(lfield, self.lelec[i], self.relec[i], 'red', linestyles='--')
                plt.hlines(rfield, self.lelec[i], self.relec[i], 'green', linestyles='--')
                plt.xlim(self.relec[i] - 15*self.dz, self.relec[i] + 15*self.dz)
                plt.vlines([self.lelec[i] + 3*self.dz,  self.relec[i] - 3*self.dz], field[:,1].min(), field[:,1].max())
                plt.show()
                raise ValueError("Left {} and right {} in-between fields are too different!".format(lfield, rfield))
            self.inbetween_fields.append(lfield)
            
    def get_lpots(self, rel_pos=0):
        """ Get potential drops on left electrode.
        
        rel_pos : float, default = 0
            Adjust electrode charge plane relative to the original wall positions. Positive values
            correspond to position closer to the IL, negative - further away from it
        """
        pots = []
        for i, field in enumerate(self.fields):
            z, field_vals = field[:,0], field[:,1]
            chg_pos = self.lelec[i] + rel_pos
            # if chg_pos is more to the left relative to real wall, use in-between field
            # oterwise simply compute potential
            if chg_pos < self.lelec[i]:
                field_vals = np.copy(field_vals)
                field_vals[(z >= chg_pos) & (z < self.lelec[i])] = self.inbetween_fields[i]
            pot_mask = (z >= chg_pos) & (z < self.no_field_points[i])
            pots.append(-np.sum(field_vals[pot_mask])*self.dz)
        return np.asarray(pots)
    
    def get_rpots(self, rel_pos=0):
        """ Get potential drop on right electrode.

        rel_pos : float, default = 0
            Adjust electrode charge plane relative to the original wall positions. Positive values
            correspond to position closer to the IL, negative - further away from it
        """
        pots = []
        for i, field in enumerate(self.fields):
            z, field_vals = field[:,0], field[:,1]
            chg_pos = self.relec[i] - rel_pos
            # if chg_pos is more to the right relative to real wall, use in-between field
            # oterwise simply compute potential
            if chg_pos > self.relec[i]:
                field_vals = np.copy(field_vals)
                field_vals[(z <= chg_pos) & (z > self.relec[i])] = self.inbetween_fields[i]
            pot_mask = (z <= chg_pos) & (z > self.no_field_points[i])
            pots.append(np.sum(field_vals[pot_mask])*self.dz)
        return np.asarray(pots)
    
    def get_pot_distribs(self, i=0):
        """ get potential distribution for i-th replica"""
        if self.pot_name:
            pot = np.loadtxt(os.path.join(self.replicas[i], self.pot_name), comments=['#','@'])
        else:
            pot_f = interpolate.InterpolatedUnivariateSpline(self.fields[i][:,0], 
                                                             self.fields[i][:,1], k=1).antiderivative()
            pot = np.c_[self.fields[i][:,0], -pot_f(self.fields[i][:,0])]
        return pot

    def plot(self, rel_pos=0, xlim=None):
        """ Plot what is going on for debug purposes."""
        lpots = self.get_lpots(rel_pos)
        rpots = self.get_rpots(rel_pos)
        fig, (ax1,ax2)  = plt.subplots(ncols=2, figsize=(8, 3))
        for i in range(len(self.replicas)):
            # left figure
            z, field = self.fields[i][:,0], self.fields[i][:,1]
            field_at_nofield_point = field[np.argmin(z - self.no_field_points[i])]
            ax1.plot(z, field)
            ax1.plot(self.no_field_points[i], field_at_nofield_point, 'ro')
            pot = self.get_pot_distribs(i)[:,1]
            # right figure
            ax2.plot(z, pot)
            ax2.hlines([lpots[i]], 0, self.relec[i], 'b')
            ax2.hlines([lpots[i] - rpots[i]], 0, self.relec[i], 'c')
            # both figures
            ax1.vlines([self.lelec[i], self.relec[i]], min(field), max(field), 'red', '--')
            ax2.vlines([self.lelec[i], self.relec[i]], min(pot), max(pot), 'red', '--')
            if xlim:
                ax1.set_xlim(*xlim)
                ax2.set_xlim(*xlim)
        ax1.set_title('field'); ax2.set_title('potential')
    
    
def mod_z_score(vals):
    """Computes modified z-score to test for outliers. Outliers are points with m-z score > 3.5
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    vals = np.asarray(vals)
    mad = np.median(np.absolute(vals - np.median(vals)))
    return np.abs(.6745*(vals - np.median(vals))/mad)

def compute_potentials(simulations, rel_pos=0, exclude_outliers=True, find_average=True):
    chgs = np.asarray([sim.surf_chg if sim else np.NaN for sim in simulations])
    chgs = np.hstack([-chgs[::-1], chgs])
    potentials = []
    for sim in simulations:    
        if sim:
            lpots = sim.get_lpots(rel_pos)
            rpots = sim.get_rpots(rel_pos)
            if exclude_outliers and np.any(mod_z_score(lpots) > 3.5):
                log_string = 'Left electrode in simulation {} contains possibly outlying potential '
                log_string += ', '.join('{:.3f}({:.1f})'.format(*vals) for vals in zip(lpots, mod_z_score(lpots)))
                log_string += '. Ignoring potentials with z-score > 3.5.'
                log.debug(log_string)
                lpots = lpots[mod_z_score(lpots) <= 3.5]
            if exclude_outliers and np.any(mod_z_score(rpots) > 3.5):
                log_string = 'Right electrode in simulation {} contains possibly outlying potential '
                log_string += ', '.join('{:.2f}({:.1f})'.format(*vals) for vals in zip(rpots, mod_z_score(rpots)))
                log_string += '. Ignoring potentials with z-score > 3.5.'
                log.debug(log_string)
                rpots = rpots[mod_z_score(rpots) <= 3.5]
            if find_average:
                potentials.append([np.nanmean(lpots), np.nanmean(rpots)])
            else:
                potentials.append([lpots, rpots])
        else:
            potentials.append([np.NaN, np.NaN])
            
    potentials = np.asarray(potentials)
    averaged_potentials = np.hstack([-potentials[:,1][::-1], -potentials[:,0]])
    #return averaged_potentials

    # get rid of NaN-s
    chgs = chgs[np.logical_not(np.isnan(averaged_potentials))]
    averaged_potentials = averaged_potentials[np.logical_not(np.isnan(averaged_potentials))]

    gromacs_data = np.c_[averaged_potentials, chgs]
    
    return gromacs_data

