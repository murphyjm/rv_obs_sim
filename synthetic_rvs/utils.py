import os
import numpy as np
import pandas as pd
from radvel.kepler import rv_drive
from radvel.utils import semi_amplitude

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.size"] = 14
from matplotlib import colors

import rvsearch
import h5py

def plot_rvs(sim, df, mp, per, outdir, fname_suffix='', red_noise_arr=None):
    '''
    Plot the RV time series (with option to show a red noise component)
    '''
    fig, ax = plt.subplots()
    
    # TODO: Fix this label string
    label_str = f'$M_\mathrm{{p}} \sin i = {mp}$ $M_\oplus$\n'
    label_str += f'$P = {per}$ d\n'
    label_str += f'RV error = {np.mean(df.errvel):.1f} m/s'
    ax.errorbar(df.time, df.mnvel, yerr=df.errvel, fmt='o', label=label_str)
    
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('RV [m/s]')
    
    x_grid = np.linspace(df.time.min(), df.time.max(), 500)
    rvtot = np.zeros(len(x_grid))
    for planet in sim.star.planets.values():
        rvtot += rv_drive(x_grid, planet.orbel, use_c_kepler_solver=False) # C solver not working for some reason

    if red_noise_arr is not None:
        ax.plot(df.time, red_noise_arr, color='k', zorder=1, label='Red Noise')

    ax.plot(x_grid, rvtot, color='r', zorder=0, label='True Keplerian')
    
    ax.legend(loc='upper right')
    
    fig.savefig(os.path.join(outdir, f'synthetic_rvs{fname_suffix}.png'), dpi=100, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_recovery_test_results(df_recoveries_fname, 
                               df_recover_test_results, 
                               outdir, 
                               mstar=0.8, 
                               plim=(1, 1000), klim=(0.1, 1000), contour_levels=[0.10, 0.5, 0.90], savefig=True, fig=None, ax=None):
    '''
    Create the plot for the injection/recovery simulation results.
    '''
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    cmap = plt.cm.bwr  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # Parameters of the colorbar 
    vmin = -5
    vmax = 5
    cbar_xticks = np.linspace(vmin, vmax, (vmax - vmin) + 1)
    bounds = cbar_xticks[cbar_xticks != 0]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar_xtick_labels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    sc = ax.scatter(df_recover_test_results['inj_period'], df_recover_test_results['inj_k'], 
                    s=20, marker='s', 
                    c=df_recover_test_results['k1_fit_minus_k1_med_baseline_over_k1_err_baseline'], 
                    cmap='bwr', norm=norm)
    
    # Calculate the completeness map
    comp = rvsearch.inject.Completeness.from_csv(df_recoveries_fname, 'inj_period', 'inj_k', mstar=mstar)
    
    # Plot contours of detection probability on the completeness plane
    pergrid, kgrid, compgrid = comp.completeness_grid(xlim=plim, ylim=klim, resolution=30)
    contours = ax.contour(pergrid, kgrid, compgrid, contour_levels, colors='k')
    
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', 
                        norm=norm, 
                        ticklocation='top', 
                        spacing='proportional', 
                        ticks=bounds, 
                        boundaries=bounds)
    
    cbar_label = f'$(K_\mathrm{{fit}} - K_\mathrm{{baseline}}) / \sigma_{{K_\mathrm{{baseline}}}}$'
    cbar.set_label(cbar_label)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks(cbar_xticks)
    cbar.ax.xaxis.set_ticklabels(cbar_xtick_labels)
    
    ax.set_facecolor('lightgray')
    ax.set_xlabel('Injected companion period [days]', labelpad=-2)
    ax.set_ylabel('Injected companion $K$ [m s$^{-1}$]')
    
    # X-axis ticks
    ax.set_xticks([1, 10, 100, 1000], minor=False) # Major ticks
    ax.set_xticklabels([1, 10, 100, 1000], fontdict={'fontsize':16}) # Major tick labels
    
    # X-axis ticks
    ax.set_yticks([0.1, 1, 10, 100, 1000], minor=False) # Major ticks
    ax.set_yticklabels([0.1, 1, 10, 100, 1000], fontdict={'fontsize':16}) # Major tick labels
    
    # Make ticks go in
    ax.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
    ax.tick_params(axis="y", direction="in", which="both", left=True, right=True)

    ax.set_xlim(plim)
    ax.set_ylim(klim)
    
    if savefig:
        fname = os.path.join(outdir, 'recoveries_test_result.png')
        fig.savefig(fname, dpi=300, facecolor='white', bbox_inches='tight')
    
    return fig, ax, contours


def plot_four_panel_recovery_test_results(parent_dir, 
                                          tel, 
                                          astro,
                                          mstar=0.8, 
                                          mplist=[2, 20],
                                          perlist=[3, 31],
                                          contour_levels=[0.10, 0.5, 0.90],
                                          savefig=True, 
                                          red_noise=False):
    '''
    Create the four-panel plot plot for the injection/recovery simulation results where the plots share the same RV instrument (internal precision) and activity level (random white noise).
    '''
    default_figsize = np.array([6.4, 4.8])
    figsize = 2 * default_figsize + np.array([1, 0]) # Make slightly wider than tall to account for colorbar
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    contours = np.empty((2,2), dtype=object)
    scatters = np.empty((2,2), dtype=object)
    
    # Colorbar stuff
    cmap = plt.cm.bwr  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    vmin = -5 
    vmax = 5
    cbar_yticks = np.linspace(vmin, vmax, (vmax - vmin) + 1)
    bounds = cbar_yticks[cbar_yticks != 0]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar_ytick_labels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    if tel == 'hires':
        rverr_inst = 1.2
    else:
        rverr_inst = 0.4
        
    if astro == 'active':
        rverr_astro = 1.2
    else:
        rverr_astro = 0.4
    test_case = f'{tel}_{astro}_{rverr_inst:.1f}_{rverr_astro:.1f}'
    output_dir = os.path.join(parent_dir, test_case)

    for i, mp in enumerate(mplist):
        for j, per in enumerate(perlist):
            
            axes[i, j].set_xscale('log')
            axes[i, j].set_yscale('log')

            run_dir = f'{mp:.1f}_{per:.1f}_{rverr_inst:.1f}_{rverr_astro:.1f}'
            if red_noise:
                run_dir += '_red_noise'
            current_dir = os.path.join(output_dir, run_dir)
            df_recoveries_fname = os.path.join(current_dir, 'recoveries.csv')

            # Extract the baseline median posterior K-amplitude value and the error
            fname = os.path.join(current_dir, 'baseline_chains.h5')
            # Open the H5 file in read mode
            with h5py.File(fname, 'r') as file:
                nchains = 8
                foo = file['0_chain']
                nsamples = foo.shape[0]
                nwalkers = foo.shape[1]
                ntot = nchains * nsamples * nwalkers
                ksamples = np.empty(ntot, dtype=float)
                # Getting the data
                for m in range(nchains):
                    data = file[f'{i}_chain']
                    ksamples[m * nsamples * nwalkers: (m+1) * nsamples * nwalkers] = data[:, :, 0].flatten()
                
            baseline_med = np.median(ksamples)
            baseline_err = np.mean(np.abs(np.quantile(ksamples, [0.16, 0.84]) - baseline_med))

            synth_rvs_fname = 'synthetic_rvs.csv'
            if red_noise:
                synth_rvs_fname = 'synthetic_rvs_w_red_noise.csv'
            synth_rvs = pd.read_csv(os.path.join(current_dir, synth_rvs_fname))
            upper_period_lim = 4 * (np.max(synth_rvs['time']) - np.min(synth_rvs['time']))
            plim = (1, upper_period_lim)
            rv_rms = np.sqrt(np.mean(np.square(synth_rvs.mnvel)))
            klim = (0.1, 10 * rv_rms)

            try:
                df_recover_test_results = pd.read_csv(os.path.join(current_dir, 'recover_test_results.csv'))
                sc = axes[i, j].scatter(df_recover_test_results['inj_period'], df_recover_test_results['inj_k'], 
                        s=20, marker='s', 
                        c=df_recover_test_results['k1_fit_minus_k1_med_baseline_over_k1_err_baseline'], 
                        cmap='bwr', norm=norm)
                scatters[i, j] = sc

                # Calculate the completeness map
                comp = rvsearch.inject.Completeness.from_csv(df_recoveries_fname, 'inj_period', 'inj_k', mstar=mstar)
                
                # Plot contours of detection probability on the completeness plane
                pergrid, kgrid, compgrid = comp.completeness_grid(xlim=plim, ylim=klim, resolution=30)
                contour = axes[i, j].contour(pergrid, kgrid, compgrid, contour_levels, colors='k')
                contours[i, j] = contour

                # Plot the known planet
                # known_planet_k = semi_amplitude(mp, per, mstar, 0, Msini_units='earth')
                axes[i, j].errorbar(per, baseline_med, yerr=baseline_err,  fmt='*', ms=20, mew=2, mec='k', color='white', ecolor='k', lw=2, capsize=5, capthick=2)

            except FileNotFoundError:
                scatters[i, j] = None
                contours[i, j] = None
            
            # Ticks and tick labels
            # X-axis ticks
            axes[i, j].set_xticks([1, 10, 100, 1000], minor=False) # Major ticks
            axes[i, j].set_xticklabels([1, 10, 100, 1000], fontdict={'fontsize':14}) # Major tick labels
            minor_xticks = axes[i, j].get_xticks(minor=True)
            axes[i, j].set_xticklabels([f'{int(tick)}' if tick in [3, 30, 300] else '' for tick in minor_xticks], fontdict={'fontsize':14}, minor=True) # Minor tick labels
            
            # Y-axis ticks
            axes[i, j].set_yticks([0.1, 1, 10, 100, 1000], minor=False) # Major ticks
            axes[i, j].set_yticklabels([0.1, 1, 10, 100, 1000], fontdict={'fontsize':14}) # Major tick labels
            minor_yticks = axes[i, j].get_yticks(minor=True)
            yfmter = lambda tick: f'{tick:.1f}' if tick < 1 else f'{int(tick):d}'
            axes[i, j].set_yticklabels([yfmter(tick) if tick in [0.2, 0.5, 2, 5, 20, 50] else '' for tick in minor_yticks], fontdict={'fontsize':14}, minor=True) # Minor tick labels

            # Axis limits
            if plim[-1] < 1000:
                axes[i, j].set_xlim(plim)
            else:
                axes[i, j].set_xlim((1, 1000))

            if klim[-1] < 1000:
                axes[i, j].set_ylim(klim)
            else:
                axes[i, j].set_ylim((0.1, 1000))

            if contours[i, j] is not None:
                fmt = {}
                countour_level_strs = np.array(contour_levels) * 100
                strs = [f'{int(clstr):d}%' for clstr in countour_level_strs] # Detection probability
                for l, s in zip(contour.levels, strs):
                    fmt[l] = s
                axes[i, j].clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=12)

            axes[i, j].set_facecolor('lightgray')
            if i != 0:
                axes[i, j].set_xlabel('Injected companion period [days]')
            if j == 0:
                axes[i, j].set_ylabel('Injected companion $K$ [m s$^{-1}$]')
            
            # Make ticks go in
            axes[i, j].tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
            axes[i, j].tick_params(axis="y", direction="in", which="both", left=True, right=True)

            if i == 0:
                axes[i, j].set_title(f'$P = {int(per)}$ days', fontsize=16)
            if j == 1:
                mp_title_str = f'$M_\\mathrm{{p}} = {int(mp)}$ $M_\\mathrm{{\\oplus}}$'
                if i == 0:
                    fig.text(0.802, 0.68, mp_title_str, va='center', rotation=-90, fontsize=16)
                else:
                    fig.text(0.802, 0.30, mp_title_str, va='center', rotation=-90, fontsize=16)

    # Colorbar stuff
    fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.83, 0.11, 0.025, 0.77])
    cbar = fig.colorbar(scatters[-1, -1], 
                        cax=cbar_ax, 
                        orientation='vertical', 
                        norm=norm, 
                        ticklocation='right', 
                        spacing='proportional', 
                        ticks=bounds, 
                        boundaries=bounds)
    cbar_label = f'$(K_\mathrm{{fit}} - K_\mathrm{{baseline}}) / \sigma_{{K_\mathrm{{baseline}}}}$'
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=16)
    cbar.ax.yaxis.set_ticks(cbar_yticks)
    cbar.ax.yaxis.set_ticklabels(cbar_ytick_labels)

    # Add a title for the whole figure
    title_str = ''
    title_str += f'$\sigma_\mathrm{{RV,inst}} = {rverr_inst:.1f}$ m s$^{{-1}}$, '
    title_str += f'$\sigma_\mathrm{{RV,astro}} = {rverr_astro:.1f}$ m s$^{{-1}}$'
    if red_noise:
        title_str += ', $+$ red noise'
    fig.suptitle(title_str, fontsize=16, x=0.465, y=0.95)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    if savefig:
        fname = os.path.join(output_dir, f'four_panel_recoveries_test_result_{tel}_{astro}')
        if red_noise:
            fname += '_red_noise'
        fig.savefig(fname + '.pdf', facecolor='white', bbox_inches='tight')
    
    return fig, axes, contours, scatters

def is_pos_def(X):
    '''
    Check if the matrix X is positive definite.
    '''
    return np.all(np.linalg.eigvals(X) > 0)

def get_red_noise(x, kernel_func, hyperparams):
    '''
    Add a red noise component to simulate stellar activity.
    '''
    n = len(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    K_tt = kernel_func(x, x, *hyperparams) + 1e-8 * np.eye(n) # Adding tiny foo factor to keep things positive definite
    assert is_pos_def(K_tt), "Covariance matrix is not positive definite"

    # Generate a multivariate Gaussian sample from the prior distribution f_* ~ N(0, K_tt)
    return np.random.multivariate_normal(np.ones(n), K_tt, 1).squeeze()

def qp_kernel(x_p, x_q, eta1, eta2, eta3, eta4):
    '''
    Quasi-periodic kernel. See e.g., Equation 1 in Kosiarek & Crossfield (2020).
    ------------
    Params: 
    x_p, x_q    ---> arrays of inputs for which to compute the covariance kernel matrix for
    Hyperparameters:
    eta1        ---> amplitude
    eta2        ---> evolutionary timescale
    eta3        ---> period
    eta4        ---> length scale of periodic component
    ------------
    Return: n x n covariance matrix relating x_p to x_q (where len(x_p) = len(x_q) = n)
    '''
    sqdist = np.sum(x_p**2,1).reshape(-1,1) + np.sum(x_q**2,1) - 2*np.dot(x_p, x_q.T)
    decay_term = sqdist / eta2**2
    periodic_term = np.sin(np.pi * np.sqrt(sqdist) / eta3)**2 / eta4**2
    return eta1**2 * np.exp(-1 * (decay_term + periodic_term))

def rbf_kernel(x_p, x_q, a, l):
    '''
    RBF kernel. See e.g., Equation 2.31 in Rasmussen & Williams (2006).
    ------------
    Params: 
    x_p, x_q ---> arrays of inputs for which to compute the covariance kernel matrix for
    a        ---> scaling prefactor (hyperparameter)
    l        ---> length scale (hyperparameter)
    ------------
    Return: n x n covariance matrix relating x_p to x_q (where len(x_p) = len(x_q) = n)
    '''
    sqdist = np.sum(x_p**2,1).reshape(-1,1) + np.sum(x_q**2,1) - 2*np.dot(x_p, x_q.T)
    return a*np.exp(-0.5 * (1/l**2) * sqdist)