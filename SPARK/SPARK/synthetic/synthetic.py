# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy import constants as const


class synth(object):
    def __init__(self, rho, T, vz):
        """
        Synthetic observation of 21cm line emission
        author: A. Marchal
        
        Parameters
        ----------
        
        Returns:
        --------
        """    
        super(synth, self).__init__()
        if len(rho.shape) == 1:
            self.dim = 1
            self.rho = np.zeros((len(rho),1,1)) *u.g*u.cm**-3
            self.T = np.zeros((len(T),1,1)) *u.K
            self.vz = np.zeros((len(vz),1,1)) *u.km*u.s**-1

            self.rho[:,0,0] = rho *u.g*u.cm**-3
            self.T[:,0,0] = T *u.K
            self.vz[:,0,0] = vz * 1.e-5 *u.km*u.s**-1
            
        else:
            self.dim = 3
            self.rho = rho *u.g*u.cm**-3
            self.T = T *u.K
            self.vz = vz * 1.e-5 *u.km*u.s**-1

        # Constant
        self.m_h = 1.6737236e-27 *u.kg
        self.C = 1.82243e18 *u.K**-1 *u.cm**-2 / (u.km * u.s**-1)
        self.box_size = 40 *u.pc
        self.resolution = 1024
        self.dz = self.box_size /self.resolution
        self.dz_cm = self.dz.to(u.cm)


    def gen(self, vmin=-40, vmax=40, dv=0.8, T_lim=[0,np.inf], thin=False):        
        # Cut temperature field 
        Tk_lim_inf = T_lim[0]
        Tk_lim_sup = T_lim[1]

        idx_phase = np.where((self.T.value > Tk_lim_inf) & (self.T.value < Tk_lim_sup))
        
        rho_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        T_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        vz_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        
        rho_cube_phase[idx_phase] = self.rho.value[idx_phase]
        T_cube_phase[idx_phase] = self.T.value[idx_phase]
        vz_cube_phase[idx_phase] = self.vz.value[idx_phase]

        # Preliminary calculation
        Delta2 = ((const.k_B.value * T_cube_phase / self.m_h.value)) * 1.e-6 #km.s-1
        n = rho_cube_phase/(self.m_h.value*1.e3)
        Delta = np.sqrt(Delta2)
        n_Delta = n / Delta

        # Spectral range
        u = np.arange(vmin,vmax+dv, dv)

        map_u = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
        for i in np.arange(T_cube_phase.shape[1]):
            for j in np.arange(T_cube_phase.shape[2]):
                map_u[:,i,j] = u

        if thin == True:
            Tb_thin_fast = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))            
            tau_thin_fast = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))

            for i in tqdm(range(len(u))):
                phi = 1. / np.sqrt(2.*np.pi) / Delta * np.exp(- (u[i] - (vz_cube_phase))**2 / (2.*Delta2))
                n_phi = n * phi
                tau_v = n_phi / T_cube_phase

                n_phi[np.where(n_phi != n_phi)] = 0.
                tau_v[tau_v != tau_v] = 0.

                Tb_thin_fast[i] = 1. / self.C.value * np.sum(n_phi,0) * self.dz_cm.value
                tau_thin_fast[i] = 1. / self.C.value * np.sum(tau_v,0) * self.dz_cm.value
            
            if self.dim == 1:
                return Tb_thin_fast[:,0,0], tau_thin_fast[:,0,0]
            else:
                return Tb_thin_fast, tau_thin_fast

        else:
            Tb = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            tau_in_front = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            
            for i in tqdm(range(T_cube_phase.shape[0])):    
                Tb_z = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
                tau_z = 1. / (self.C.value * np.sqrt(2.*np.pi)) * n_Delta[i] / T_cube_phase[i] * np.exp(- (map_u - (vz_cube_phase[i]))**2 
                                                                                               / (2.*Delta2[i])) * self.dz_cm.value
                idx_nonzero = ~np.isnan(tau_z[0])
                Tb_z[:,idx_nonzero] = T_cube_phase[i,idx_nonzero] * (1. - np.exp(-1.*tau_z[:,idx_nonzero])) * np.exp(-1.*tau_in_front[:,idx_nonzero])
                
                tau_in_front[:,idx_nonzero] += tau_z[:,idx_nonzero]
                
                Tb += Tb_z 
            if self.dim == 1:
                return Tb[:,0,0], tau_in_front[:,0,0]
            else:
                return Tb, tau_in_front


if __name__ == '__main__':    
    # Open data 
    path = '/home/amarchal/SPARK/data/'
    
    hdu_list_rho = fits.open(path + 'rho_cube_sample.fits')
    hdu_list_T = fits.open(path + 'T_cube_sample.fits')
    hdu_list_vz = fits.open(path + 'vz_cube_sample.fits')
    
    #Velocity range and channel spacing
    vmin = -40 #km.s-1
    vmax = 40 #km.s-1
    dv = 0.8 #km.s-1
    
    rho_cube = hdu_list_rho[0].data[:,0,0] #g.cm-3
    T_cube = hdu_list_T[0].data[:,0,0] #K
    vz_cube = hdu_list_vz[0].data[:,0,0] #cm.s-1

    core = synth(rho=rho_cube, T=T_cube, vz=vz_cube)
    cube, tau = core.gen(vmin=-40, vmax=40, dv=0.8, thin=False)
    cube_thin, tau_thin = core.gen(vmin=-40, vmax=40, dv=0.8, thin=True)
