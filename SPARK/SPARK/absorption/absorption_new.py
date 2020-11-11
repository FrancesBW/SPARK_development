# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy import optimize 

class lbfgs_abs(object):
    def __init__(self, Tb, tau, rms_Tb=None, rms_tau=None, hdr=None):
        """
        Joint fit emission and absorption spectra for GASKAP collaboration
        author: A. Marchal
        
        Edited by F. Buckland-Willis
        
        Parameters
        ----------
        
        Returns:
        --------
        """    
        super(lbfgs_abs, self).__init__()
        self.Tb = Tb
        self.tau = tau
        self.rms_Tb = rms_Tb if rms_Tb is not None else 1.
        self.rms_tau = rms_tau if rms_tau is not None else 1.
        self.hdr = hdr if hdr is not None else None
        if self.hdr is not None : self.v = self.mean2vel(self.hdr["CRVAL3"]*1.e-3, self.hdr["CDELT3"]*1.e-3, 
                                                         self.hdr["CRPIX3"], np.arange(len(self.tau)))

    def run(self, n_gauss=18, lb_amp=0, ub_amp=100, lb_mu=1, ub_mu=500, lb_sig=1, ub_sig=100, lambda_Tb=1, 
            lambda_tau=1, lambda_mu=100, lambda_sig=100, iprint_init=1, amp_fact_init=0.666, sig_init=2., 
            maxiter=15000, maxiter_init=15000, iprint=1):

        #Flag test basic properties
        if len(self.Tb) != len(self.tau) : 
            print("Emission and absorption spectra must have the same size.")
            sys.exit()
    
        #Dimensions
        dim_v = len(self.Tb)

        #RMS cube format 
        rms = [self.rms_Tb,self.rms_tau]
        
        #initialise parameter divinding up so it's 50/50 with more being taken from emission if odd number
        
        if n_gauss==1:
            params_Tb = self.init_spectrum(np.full((3),1.), 1, self.Tb, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
        else:
            if n_gauss%2==0:
                n_gauss_Tb=int(n_gauss/2)
                n_gauss_tau=int(n_gauss/2)
            else:
                n_gauss_Tb=int(n_gauss//2+1)
                n_gauss_tau=int(n_gauss//2)
            params_Tb = self.init_spectrum(np.full((3*n_gauss_Tb),1.), n_gauss_Tb, self.Tb, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
            params_tau = self.init_spectrum(np.full((int(3*n_gauss_tau)),1.), n_gauss_tau, self.tau, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
         
        #Allocate and init arrays
        cube = np.moveaxis(np.array([self.Tb,self.tau]),0,1)
        params = np.full((3*n_gauss,2),1.)
        bounds = self.init_bounds(cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
        
        #Copy result as init of the absorption spectrum
        if n_gauss==1:
            params[:,0] = params_Tb
            params[:,1] = params_Tb
        else:
            params[:3*n_gauss_Tb,0] = params_Tb
            params[:3*n_gauss_Tb,1] = params_Tb
            params[3*n_gauss_Tb:,0] = params_tau
            params[3*n_gauss_Tb:,1] = params_tau

        #Update both with regularization
        result = optimize.fmin_l_bfgs_b(self.f_g, params.ravel(), args=(n_gauss, cube, rms, lambda_Tb, lambda_tau, 
                                                                   lambda_mu, lambda_sig), 
                                        bounds=bounds, approx_grad=False, disp=iprint, maxiter=maxiter)
        
        # params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))   

        return result

    def order_params(self, params):
    #This function will order the parameters by their gaussian centres
        x_order=np.sort(params[1:-1:3])
        ordered=np.array([])
        for item in x_order:
            idx=np.where(params==item)
            gauss=[params[idx[0][0]-1],params[idx[0][0]],params[idx[0][0]+1]]
            ordered=np.concatenate((ordered,gauss),axis=0)
        return ordered

    def mean2vel(self, CRVAL, CDELT, CRPIX, mean):
        return [(CRVAL + CDELT * (mean[i] - CRPIX)) for i in range(len(mean))]       


    def init_bounds_spectrum(self, n_gauss, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig):
        bounds_inf = np.zeros(3*n_gauss)
        bounds_sup = np.zeros(3*n_gauss)
        
        bounds_inf[0::3] = lb_amp
        bounds_inf[1::3] = lb_mu
        bounds_inf[2::3] = lb_sig
        
        bounds_sup[0::3] = ub_amp
        bounds_sup[1::3] = ub_mu
        bounds_sup[2::3] = ub_sig
        
        return [(bounds_inf[i], bounds_sup[i]) for i in np.arange(len(bounds_sup))]


    def init_bounds(self, cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig):
        bounds_inf = np.zeros(params.shape)
        bounds_sup = np.zeros(params.shape)

        for i in range(cube.shape[1]):
            for k in np.arange(params.shape[0]/3):
                bounds_A = [lb_amp, ub_amp]
                bounds_mu = [lb_mu, ub_mu]
                bounds_ampma = [lb_sig, ub_sig]

                bounds_inf[int(0+(3*k)),i] = bounds_A[0]
                bounds_inf[int(1+(3*k)),i] = bounds_mu[0]
                bounds_inf[int(2+(3*k)),i] = bounds_ampma[0]
                
                bounds_sup[int(0+(3*k)),i] = bounds_A[1]
                bounds_sup[int(1+(3*k)),i] = bounds_mu[1]
                bounds_sup[int(2+(3*k)),i] = bounds_ampma[1]
            
        return [(bounds_inf.ravel()[i], bounds_sup.ravel()[i]) for i in np.arange(len(bounds_sup.ravel()))]


    def f_g(self, pars, n_gauss, data, rms, lambda_Tb, lambda_tau, lambda_mu, lambda_sig):
        params = np.reshape(pars, (3*n_gauss, data.shape[1]))
        
        x = np.arange(data.shape[0])
        
        model = np.zeros(data.shape)
        dF_over_dB = np.zeros((params.shape[0], data.shape[0], data.shape[1]))
        product = np.zeros((params.shape[0], data.shape[0], data.shape[1]))
        deriv = np.zeros((params.shape[0], data.shape[1]))

        for i in np.arange(data.shape[1]):
            for k in np.arange(n_gauss):
                model[:,i] += self.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], params[2+(k*3),i])
                
                dF_over_dB[0+(k*3),:,i] += (1. 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))
                dF_over_dB[1+(k*3),:,i] += (params[0+(k*3),i] * (x - params[1+(k*3),i]) / (params[2+(k*3),i])**2 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))
                dF_over_dB[2+(k*3),:,i] += (params[0+(k*3),i] * (x - params[1+(k*3),i])**2 / (params[2+(k*3),i])**3 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))                

        F = model - data   
        
        F[:,0] /= rms[0]
        F[:,1] /= rms[1]
        
        for i in np.arange(data.shape[1]):
            for v in np.arange(data.shape[0]):
                if i ==0 :
                    product[:,v,i] = lambda_Tb * dF_over_dB[:,v,i] * F[v,i]
                else:
                    product[:,v,i] = lambda_tau * dF_over_dB[:,v,i] * F[v,i]

                        
        deriv = np.sum(product, axis=1)

        J =  0.5 * lambda_Tb * np.sum(F[:,0]**2) + 0.5 * lambda_tau * np.sum(F[:,1]**2)

        R_mu = 0.5 * lambda_mu * np.sum((params[1::3,1] / params[1::3,0] - 1.)**2)
        R_sig = 0.5 * lambda_sig * np.sum((params[2::3,1] / params[2::3,0] - 1.)**2)
        
        deriv[1::3,0] = deriv[1::3,0] - (lambda_mu * params[1::3,1] / params[1::3,0]**2.
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[1::3,1] = deriv[1::3,1] + (lambda_mu / params[1::3,0]
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[2::3,0] = deriv[2::3,0] - (lambda_sig *  params[2::3,1] / params[2::3,0]**2.
                                         * (params[2::3,1] / params[2::3,0] - 1.))
        
        deriv[2::3,1] = deriv[2::3,1] + (lambda_sig / params[2::3,0]
                                         * (params[2::3,1] / params[2::3,0] - 1.))
        
        return J + R_mu + R_sig, deriv.ravel()

    
    def f_g_spectrum(self, params, n_gauss, data):
        x = np.arange(data.shape[0])
        
        model = np.zeros(data.shape[0])
        dF_over_dB = np.zeros((params.shape[0], data.shape[0]))
        product = np.zeros((params.shape[0], data.shape[0]))
        deriv = np.zeros((params.shape[0]))
        
        for k in np.arange(n_gauss):
            model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])
            
            dF_over_dB[0+(k*3),:] += (1. 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))
            dF_over_dB[1+(k*3),:] += (params[0+(k*3)] * (x - params[1+(k*3)]) / (params[2+(k*3)])**2 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))
            dF_over_dB[2+(k*3),:] += (params[0+(k*3)] * (x - params[1+(k*3)])**2 / (params[2+(k*3)])**3 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))                
    
        F = model - data                
                
        for v in np.arange(data.shape[0]):
            product[:,v] = dF_over_dB[:,v] * F[v]
                        
        deriv = np.sum(product, axis=1)
        
        J = 0.5*np.sum(F**2)
        
        return J, deriv.ravel()


    def init_spectrum(self, params, n_gauss, data, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, 
                      iprint, amp_fact_init, sig_init, maxiter):


        for i in np.arange(int(n_gauss)):
            n = i+1
            x = np.arange(data.shape[0])
            bounds = self.init_bounds_spectrum(n, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
            model = np.zeros(data.shape[0])
            
            for k in np.arange(n):
                model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])

            residual = model - data
            xx = np.zeros((3*n,1))
            for p in np.arange(3*n):
                xx[p] = params[p]
                
            xx[1+(i*3)] = np.where(residual == np.min(residual))[0][0]   
            xx[0+(i*3)] = data[int(xx[1+(i*3)])] * amp_fact_init
            xx[2+(i*3)] = sig_init

            result = optimize.fmin_l_bfgs_b(self.f_g_spectrum, xx, args=(n, data), 
                                        bounds=bounds, approx_grad=False, disp=iprint, maxiter=maxiter)
    
            for p in np.arange(3*n):
                params[p] = result[0][p]
            
        return params


    def gaussian(self, x, amp, mu, sig):
        return amp * np.exp(-((x - mu)**2)/(2. * sig**2))


    def model_spectrum(self, params, data, n_gauss):
        x = np.arange(data.shape[0])
        model = np.zeros(len(x))        
        
        for k in np.arange(n_gauss):
            model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])

        return model


    def model(self, params, data, n_gauss):
        x = np.arange(data.shape[0])
        model = np.zeros(data.shape)
        
        for i in np.arange(data.shape[1]):
            for k in np.arange(n_gauss):
                model[:,i] += self.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], params[2+(k*3),i])
            
        return model


if __name__ == '__main__':    
    print("lbfgs_abs module")
    core = lbfgs_abs(np.zeros(30), np.zeros(30))
    

