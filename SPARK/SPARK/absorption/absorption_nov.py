# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy import optimize 
from scipy.stats import pearsonr as pcc
import copy
import matplotlib.pyplot as plt

class lbfgs_abs(object):
    def __init__(self, Tb, tau, rms_Tb=1., rms_tau=1., lambda_Tb=1., lambda_tau=1., lambda_mu=1., lambda_sig=1., hdr=None):
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
        self.rms_Tb = rms_Tb
        self.rms_tau = rms_tau
        self.lambda_Tb = lambda_Tb
        self.lambda_tau = lambda_tau
        self.lambda_mu = lambda_mu
        self.lambda_sig = lambda_sig
        self.hdr = hdr
        if self.hdr is not None : self.v = self.mean2vel(self.hdr["CRVAL3"]*1.e-3, self.hdr["CDELT3"]*1.e-3, 
                                                         self.hdr["CRPIX3"], np.arange(len(self.tau)))

    def run(self, n_gauss=18, lb_amp=0, ub_amp=100, lb_mu=1, ub_mu=500, lb_sig=1, ub_sig=100, pcc_mu=0.9, pcc_sig=0.9, red_chi_sq_thres=1.5, amp_fact_init=0.666, sig_init=2., maxiter=15000, maxiter_init=15000, max_cor_iter=10, iprint=1, iprint_init=1, init=0, prior=None):

        #Flag test basic properties
        if len(self.Tb) != len(self.tau) : 
            print("Emission and absorption spectra must have the same size.")
            sys.exit()
            
        #convert the emission and the absorption to similar ranges by calculating a normalisation factor
        norm_factor=np.nanmax(self.tau)/np.nanmax(self.Tb)
        self.Tb=self.Tb*norm_factor
        self.rms_Tb=self.rms_Tb*norm_factor
    
        #Dimensions
        dim_v = len(self.Tb)

        #RMS cube format 
        rms = np.transpose(np.array([self.rms_Tb,self.rms_tau]))
        
        #first we retain the old fresh initialisation:
        if init==2:
            params_Tb, params_tau = self.fresh_init(n_gauss, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
         
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
            prelim_result = optimize.fmin_l_bfgs_b(self.f_g, params.ravel(), args=(n_gauss, cube, rms), bounds=bounds, 
                                                   approx_grad=False, disp=iprint, maxiter=maxiter)

        #now we write the case where we do an iterative solution we will set up it separately for if we need to start from one 
        #or from a predefined solution
        else:
            #this condition is for adding to a solution of lower number of gaussians
            cube=np.moveaxis(np.array([self.Tb,self.tau]),0,1)
            if init==1:
                #catch invalid files
                try:
                    params=np.loadtxt(prior)
                except (OSError,ValueError) as e:
                    print('Invalid file path provided for prior')
                    sys.exit()
                #params need to be multiples of three, as there are 3 components for every gaussian
                if len(params)%3!=0:
                    print('Amount of priors not divisible by 3, please check and try again')
                    sys.exit()
                n_gauss_fitted=int(len(params)/3.)
            
            if init==0:
                n_gauss_fitted=1
                #we need to initialise and optimise the fit for 1 gaussian before doing the iterative solution , as it will 
                #break otherwise
                params = np.full((3,2),1.)
                params_Tb, params_tau = self.fresh_init(n_gauss_fitted, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
                params[:,0] = params_Tb
                params[:,1] = params_Tb
                bounds = self.init_bounds(cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
                params=self.self_correcting_optimisation(cube, rms, params, n_gauss_fitted, lb_amp, ub_amp, lb_mu, ub_mu, 
                                                         lb_sig, ub_sig, pcc_mu, pcc_sig, red_chi_sq_thres, iprint_init, 
                                                         amp_fact_init, sig_init, maxiter, maxiter_init, max_cor_iter, iprint, 
                                                         bounds)
            
            #now we move onto the iterative gaussian process for the fitting
            while n_gauss_fitted<n_gauss:
                print('a loop in the while')
                current_model=self.model(params, cube, n_gauss_fitted)
                self.plot_model(cube, current_model, rms)
                res_cube=cube-current_model
                #find the position of the minimum value across both spectra
                min_idx=np.where(res_cube==np.nanmax(res_cube))[0][0]
                amp_min=np.abs(res_cube[min_idx])
                #add these onto the end of the params list in the order: amp, mu, sig
                params=np.append(params,[amp_min],axis=0)
                params=np.append(params,[[min_idx,min_idx]],axis=0)
                params=np.append(params,[[sig_init,sig_init]],axis=0)
                n_gauss_fitted+=1
                bounds = self.init_bounds(cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
                self.reset_lambdas()
                params=self.self_correcting_optimisation(cube, rms, params, n_gauss_fitted, lb_amp, ub_amp, lb_mu, ub_mu, 
                                                         lb_sig, ub_sig, pcc_mu, pcc_sig, red_chi_sq_thres, iprint_init, 
                                                         amp_fact_init, sig_init, maxiter, maxiter_init, max_cor_iter, iprint, 
                                                         bounds)       
                    
        #the prelim_result needs to be converted to physical quantities, as it is just pixel quantities at this point, we can also undo the normalisation at this point
        result=copy.copy(params)
        result[0::6]=result[0::6]/norm_factor
        return result

    def self_correcting_optimisation(self, cube, rms, params, n_gauss, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, pcc_mu, pcc_sig, red_chi_sq_thres, iprint_init, amp_fact_init, sig_init, maxiter, maxiter_init, max_cor_iter, iprint, bounds):
        metric_limits=np.array([red_chi_sq_thres, red_chi_sq_thres, pcc_mu, pcc_sig])
        print('Optimising for {} gaussians'.format(str(n_gauss)))
        for i in np.arange(max_cor_iter):
            print('**********************************************')
            print('Parameters prior to optimisation lbfgs')
            print(params)
            cal=self.f_g(params, n_gauss, cube, rms)
            print('The J value of this guess is:. {}'.format(str(cal[0])))
            result=optimize.fmin_l_bfgs_b(self.f_g, params.ravel(), args=(n_gauss, cube, rms), bounds=bounds,
                                          approx_grad=False, disp=iprint, maxiter=maxiter)    
            new_params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))
            if np.all(new_params==params):
                print("No improvement found, exiting correction stage")
                break
            else:
                params=new_params
            print('Parameters after the optimisation')
            print(params)
            print('J value {}'.format(str(result[1])))
            print('**********************************************')
            model_cube=self.model(params, cube, n_gauss)
            fit_metrics=self.reduced_chi_squared(cube, model_cube, rms, n_gauss)
            try:
                corr_metrics=np.array([pcc(params[1::3,0],params[1::3,1])[0], pcc(params[2::3,0],params[2::3,1])[0]])
            except ValueError:
                corr_metrics=np.ones(2)
            good_fit=True
            print('Calculated values are:')
            print(fit_metrics)
            print(corr_metrics)
            #normalise the lambda values so that the smallest is equal to 1, preserves ratio whilst not allowing them to get too 
            #big
            self.normalise_lambdas()
            if fit_metrics[0]>metric_limits[0]: 
                print('emission not fitted enough, current lambda_tb = {}'.format(str(self.lambda_Tb)))
                print('This is iteration {}'.format(str(i)))
                self.lambda_Tb+=10/max_cor_iter
                print('ammended lambda_tb = {}'.format(str(self.lambda_Tb)))
                good_fit=False
            if fit_metrics[1]>metric_limits[1]: 
                print('absorption not fitted enough, current lambda_tau = {}'.format(str(self.lambda_tau)))
                self.lambda_tau+=10/max_cor_iter
                print('ammended lambda_tau = {}'.format(str(self.lambda_tau)))
                good_fit=False
            if corr_metrics[0]<metric_limits[2]: 
                print('mu dont align enough, current lambda_mu = {}'.format(str(self.lambda_mu)))
                self.lambda_mu+=10/max_cor_iter
                print('ammended lambda_mu = {}'.format(str(self.lambda_mu)))
                good_fit=False
            if corr_metrics[1]<metric_limits[3]: 
                print('sig dont align enough, current lambda_sig = {}'.format(str(self.lambda_sig)))
                self.lambda_sig+=10/max_cor_iter
                print('ammended lambda_sig = {}'.format(str(self.lambda_sig)))
                good_fit=False
            if good_fit: 
                print('broke early')
                return params
                break
            else:
                print('pushed {} time(s)'.format(str(i+1)))
                
        return params
    
    def reduced_chi_squared(self, cube, model_cube, rms, n_gauss):
        res=model_cube-cube
        chi_sq=res/rms
        red_chi_sq_Tb=np.sum(chi_sq[:,0]**2)/(len(cube)-3*n_gauss)
        red_chi_sq_tau=np.sum(chi_sq[:,1]**2)/(len(cube)-3*n_gauss)
        return [red_chi_sq_Tb, red_chi_sq_tau]
            
    def normalise_lambdas(self):
        max_lambda=np.nanmin(np.array([self.lambda_Tb, self.lambda_tau, self.lambda_mu, self.lambda_sig]))
        self.lambda_Tb=self.lambda_Tb/max_lambda
        self.lambda_tau=self.lambda_tau/max_lambda
        self.lambda_mu=self.lambda_mu/max_lambda
        self.lambda_sig=self.lambda_sig/max_lambda
        return
    
    def reset_lambdas(self):
        self.lambda_Tb=1.
        self.lambda_tau=1.
        self.lambda_mu=1.
        self.lambda_sig=1.
        return 
    
    
    def fresh_init(self, n_gauss, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init):
        if n_gauss==1:
            params_Tb = self.init_spectrum(np.full((3),1.), 1, self.Tb, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
            return params_Tb, None
        else:
            if n_gauss%2==0:
                n_gauss_Tb=int(n_gauss/2)
                n_gauss_tau=int(n_gauss/2)
            else:
                n_gauss_Tb=int(n_gauss//2+1)
                n_gauss_tau=int(n_gauss//2)
            params_Tb = self.init_spectrum(np.full((3*n_gauss_Tb),1.), n_gauss_Tb, self.Tb, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
            params_tau = self.init_spectrum(np.full((int(3*n_gauss_tau)),1.), n_gauss_tau, self.tau, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, maxiter_init)
            return params_Tb, params_tau

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


    def f_g(self, pars, n_gauss, data, rms):
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
        
        F /= rms
        
        for i in np.arange(data.shape[1]):
            for v in np.arange(data.shape[0]):
                if i ==0 :
                    product[:,v,i] = self.lambda_Tb * dF_over_dB[:,v,i] * F[v,i]
                else:
                    product[:,v,i] = self.lambda_tau * dF_over_dB[:,v,i] * F[v,i]

                        
        deriv = np.sum(product, axis=1)

        J =  0.5 * self.lambda_Tb * np.sum(F[:,0]**2) + 0.5 * self.lambda_tau * np.sum(F[:,1]**2)

        R_mu = 0.5 * self.lambda_mu * np.sum((params[1::3,1] / params[1::3,0] - 1.)**2)
        R_sig = 0.5 * self.lambda_sig * np.sum((params[2::3,1] / params[2::3,0] - 1.)**2)
        
        deriv[1::3,0] = deriv[1::3,0] - (self.lambda_mu * params[1::3,1] / params[1::3,0]**2.
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[1::3,1] = deriv[1::3,1] + (self.lambda_mu / params[1::3,0]
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[2::3,0] = deriv[2::3,0] - (self.lambda_sig *  params[2::3,1] / params[2::3,0]**2.
                                         * (params[2::3,1] / params[2::3,0] - 1.))
        
        deriv[2::3,1] = deriv[2::3,1] + (self.lambda_sig / params[2::3,0]
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
                
            xx[1+(i*3)] = np.where(residual == np.nanmin(residual))[0][0]
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

    def plot_model(self, cube, model_cube, rms):
        v=np.arange(len(cube))
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=(20,16),gridspec_kw={'height_ratios': [4,1,1,4]})
        fig.subplots_adjust(hspace=0.)
        x = np.arange(cube.shape[0])
        ax1.step(v, cube[:,0], color='cornflowerblue', linewidth=2.)
        ax1.plot(v, model_cube[:,0], color='k')
        ax2.plot(v, -model_cube[:,0] + cube[:,0], color='k')
        ax2.fill_between(v, -3.*rms[:,0], 3.*rms[:,0], facecolor='lightgray', color='lightgray')
        ax3.plot(v, -model_cube[:,1] + cube[:,1], color='k')
        ax3.fill_between(v, -3.*rms[:,1], 3.*rms[:,1], facecolor='lightgray', color='lightgray')
        ax4.step(v, -cube[:,1], color='cornflowerblue', linewidth=2.)
        ax4.plot(v, -model_cube[:,1], color='k')
        ax1.set_ylabel(r'T$_{B}$ [K]', fontsize=16)
        ax2.set_ylabel(r'$T_B$ res [K]', fontsize=12)
        ax3.set_ylabel(r"$\tau$ res", fontsize=12)
        ax4.set_ylabel(r'$- \tau$', fontsize=16)
        ax4.set_xlabel(r'v [km s$^{-1}$]', fontsize=16)
        plt.show()
        return

if __name__ == '__main__':    
    print("lbfgs_abs module")
    core = lbfgs_abs(np.zeros(30), np.zeros(30))
    

