import numpy as np
from astropy.io import fits
import astropy.table as pytabs
import matplotlib.pyplot as plt
import os
import time

#os.chdir('SPARK/SPARK/absorption/')
#print(os.getcwd())
from SPARK.SPARK.absorption.absorption_new import lbfgs_abs
#os.chdir('../../../')


test_sources=['J2232','3C433','4C33.48','PKS1607','4C32.44','3C225B','3C154','3C133','3C111B','3C041A']
path=''
cat = fits.getdata(path+"all_sponge_sources_table_tighter.fits")
data_s = pytabs.Table(cat)
value_cube=[]
save_path='exploration_output/'
values=[1,10,20,30,40,50,60,70,80,90,100]
global_start_time=time.perf_counter()
local_start_time=time.perf_counter()
local_end_time=0

for source in test_sources:
    local_start_time+=local_end_time
    print('started source {}'.format(source))
    source_grid=np.zeros((11,11,11,11))
    idx_absline=np.where(data_s["NAMES"]==source)[0][0]
    #idx_absline = np.random.randint(57)
    nan_values=np.isnan(data_s[idx_absline]["VEL"])
    numeric_values=[not i for i in nan_values]
    v = data_s[idx_absline]["VEL"][numeric_values]
    #correct velocities to model over
    chiller_vel_idx=np.intersect1d(np.where(v>-100.),np.where(v<80.))
    
    #initialise the reduced data
    v = v[chiller_vel_idx]
    Tb = data_s[idx_absline]["TB"][numeric_values][chiller_vel_idx]
    tau = data_s[idx_absline]["TAU"][numeric_values][chiller_vel_idx]
    rms_Tb=data_s[idx_absline]["SIG_TB"][numeric_values][chiller_vel_idx]
    rms_tau=data_s[idx_absline]["SIG_TAU"][numeric_values][chiller_vel_idx]
    
    #Channel spacing
    dv = np.diff(v)[0]
    
    #hdr
    hdr=fits.Header()
    hdr["CDELT3"] = dv
    hdr["CRPIX3"] = 0
    hdr["CRVAL3"] = v[0]*1.e3
    
    #parameters
    amp_fact_init = 2./3.
    sig_init = 2.
    iprint_init = -1
    iprint = -1
    maxiter_init = 15000
    maxiter = 15000
    
    for m in np.arange(11):
        for n in np.arange(11):
            for o in np.arange(11):
                for p in np.arange(11):
                    n_gauss = 6             #@param {type:"slider", min:3, max:24, step:3}
                    prefix=save_path+"{}_Tb{}_tau{}_mu{}_sig{}_gaussian6".format(source,str(m),str(n),str(o),str(p))
                    lambda_Tb = values[m]           #@param {type:"slider", min:1, max:100, step:10}
                    lambda_tau = values[n]          #@param {type:"slider", min:1, max:100, step:10}
                    lambda_mu = values[o]           #@param {type:"slider", min:0, max:100, step:10}
                    lambda_sig = values[p]          #@param {type:"slider", min:0, max:100, step:10}
                    lb_amp = 0.
                    ub_amp = np.max(Tb)
                    lb_mu = 1
                    ub_mu = len(tau)
                    lb_sig= 1
                    ub_sig = 100
            
                    core = lbfgs_abs(Tb=Tb, tau=tau, hdr=hdr)
            
                    result = core.run(n_gauss=n_gauss,
                              lb_amp=lb_amp,
                              ub_amp=ub_amp,
                              lb_mu=lb_mu,
                              ub_mu=ub_mu,
                              lb_sig=lb_sig,
                              ub_sig=ub_sig,
                              lambda_Tb=lambda_Tb,
                              lambda_tau=lambda_tau,
                              lambda_mu=lambda_mu,
                              lambda_sig=lambda_sig,
                              amp_fact_init=amp_fact_init,
                              sig_init=sig_init,
                              maxiter=maxiter,
                              maxiter_init=maxiter_init,
                              iprint=iprint,
                              iprint_init=iprint_init)
                
                    #Compute model
                    cube = np.moveaxis(np.array([Tb,tau]),0,1)
                    params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))
                    model_cube = core.model(params, cube, n_gauss)
                    res_Tb=np.abs(model_cube[:,0] - Tb)
                    res_tau=np.abs(model_cube[:,1] - tau)
                              
                    cube = np.moveaxis(np.array([Tb,tau]),0,1)
                    params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))
                    vfield_Tb = core.mean2vel(hdr["CRVAL3"]*1.e-3, hdr["CDELT3"], hdr["CRPIX3"], params[1::3,0])
                    vfield_tau = core.mean2vel(hdr["CRVAL3"]*1.e-3, hdr["CDELT3"], hdr["CRPIX3"], params[1::3,1])
                              
                    model_cube = core.model(params, cube, n_gauss)
                              
                    #do the plotting
                    plt.figure(figsize=(16,6))
                              
                    plt.subplot(1,2,1)
                    plt.plot(vfield_Tb, vfield_tau, "+b")
                    plt.xlabel(r"v$_{\rm Tb}$ [km s$^{-1}$]", fontsize=16)
                    plt.ylabel(r"v$_{\rm tau}$ [km s$^{-1}$]", fontsize=16)
                              
                    plt.subplot(1,2,2)
                    plt.plot(params[:,0][2::3]*dv, params[:,1][2::3]*dv, "+r")
                    plt.xlabel(r"$\sigma_{\rm Tb}$ [km s$^{-1}$]", fontsize=16)
                    plt.ylabel(r"$\sigma_{\rm tau}$ [km s$^{-1}$]", fontsize=16)
                    plt.savefig(prefix+"correlation.png", format='png', bbox_inches='tight', pad_inches=0.02)
                    plt.close()
                              
                    plt.figure(figsize=(8,6))
                    plt.subplot(1,1,1)
                    plt.plot(vfield_Tb, params[:,0][2::3]*dv, "+g")
                    plt.plot(vfield_tau, params[:,1][2::3]*dv, "+m")
                    plt.xlabel(r"v [km s$^{-1}$]", fontsize=16)
                    plt.ylabel(r"$\sigma$ [km s$^{-1}$]", fontsize=16)
                    plt.savefig(prefix+"sigma_v.png", format='png', bbox_inches='tight', pad_inches=0.02)
                    plt.close()
                              
                    #bigger plot
                    pvalues = np.logspace(-1, 0, n_gauss)
                    pmin = pvalues[0]
                    pmax = pvalues[-1]
                              
                              
                    def norm(pval):
                        return (pval - pmin) / float(pmax - pmin)
                              
                    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=(20,16),gridspec_kw={'height_ratios': [4,1,1,4]})
                    fig.subplots_adjust(hspace=0.)
                    x = np.arange(cube.shape[0])
                    ax1.step(v, cube[:,0], color='cornflowerblue', linewidth=2.)
                    ax1.plot(v, model_cube[:,0], color='k')
                    ax2.plot(v, model_cube[:,0] - Tb, color='k')
                    ax2.fill_between(v, -3.*rms_Tb, 3.*rms_Tb, facecolor='lightgray', color='lightgray')
                    ax3.plot(v, -model_cube[:,1] + tau, color='k')
                    ax3.fill_between(v, -3.*rms_tau, 3.*rms_tau, facecolor='lightgray', color='lightgray')
                    ax4.step(v, -cube[:,1], color='cornflowerblue', linewidth=2.)
                    ax4.plot(v, -model_cube[:,1], color='k')
                    for j in np.arange(cube.shape[1]):
                        for k in np.arange(n_gauss):
                            line = core.gaussian(x, params[0+(k*3),j], params[1+(k*3),j], params[2+(k*3),j])
                            if j == 1:
                                ax4.plot(v, -line, color=plt.cm.rainbow(pvalues[k]), linewidth=2.)
                            else:
                                ax1.plot(v, line, color=plt.cm.rainbow(pvalues[k]), linewidth=2.)

                    ax1.set_ylabel(r'T$_{B}$ [K]', fontsize=16)
                    ax2.set_ylabel(r'$T_B$ res [K]', fontsize=12)
                    ax3.set_ylabel(r"$\tau$ res", fontsize=12)
                    ax4.set_ylabel(r'$- \tau$', fontsize=16)
                    ax4.set_xlabel(r'v [km s$^{-1}$]', fontsize=16)
                    ax1.text(20,10,'J= {}'.format(result[1]), fontsize=16)
                    plt.savefig(prefix+"result_spectra.png", format='png', bbox_inches='tight', pad_inches=0.02)
                    plt.close()
            
                    np.savetxt(prefix+'params.txt',params)
                    print('finished tb_{}, tau_{}, mu_{}, sig_{}'.format(str(m),str(n),str(o),str(p)))
    time_elapsed=time.perf_counter()-local_start_time
    local_end_time+=time_elapsed
    print('finished source {} in {} minutes'.format(source, int(time_elapsed/60.)))
total_time_elapsed=local_end_time-global_start_time
print('Total exploration completed in {} hours'.format(int(total_time_elapsed/3600.)))
