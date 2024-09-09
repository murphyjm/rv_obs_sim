import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from rv_obs_sim.synthSim import SynthSim, RVStar, RVPlanet

from utils import plot_rvs, plot_recovery_test_results, get_red_noise, qp_kernel

import radvel
from radvel.plot.orbit_plots import MultipanelPlot

import rvsearch

import math

###########################
##### Hyperparameters #####

# Orbit of the known planet
e = 0
omega = np.pi/2
obs_bjd_start = 0

# Limits of the injection recovery grid for RV search
plim = (1, 1000) # Period grid limits. Days
klim = (0.1, 1000) # K-amplitude grid limits. m/s

###########################
###########################

def parse_args():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Run the simulation code for specified scenario.')
    parser.add_argument('mpsini', type=float, help="The Mp sini of the known planet in the system in Earth units.")
    parser.add_argument('period', type=float, help="The orbital period of the known planet in the system in days.")
    parser.add_argument('rverr_internal', type=float, help="The internal precision of the RV instrument in m/s.")
    parser.add_argument('--rverr_astro', type=float, default=0.4, help="The astrophysical jitter to add to the RVs in m/s.")
    parser.add_argument('--recoveries_fname', type=str, default=None, help="Path to the recoveries object if injection recovery tests have already been run.")
    parser.add_argument('--mstar', type=float, default=0.8, help="The mass of the host star in solar units.")
    parser.add_argument('--nobs', type=int, default=40, help="The number of RVs to include in the simulation.")
    parser.add_argument('--weather_losses', type=float, default=0.3, help="Simulate gaps in the observations from some fraction of weather losses.")
    parser.add_argument('--obs_cadence', type=float, default=3, help="The number of observations to obtain per orbital period of the known planet. E.g., obs_cadence = 3 for a planet with P = 10 d means take 3 RVs every 10 days.")
    parser.add_argument('--time_jit', type=float, default=0.333, help="The jitter to add to the time grid in units of days.")
    parser.add_argument('--num_sim', type=int, default=1000, help="The number of injection/recovery trials to conduct.")
    parser.add_argument('--num_cpus', type=int, default=os.cpu_count(), help="The number of CPUs to use for the RVSearch injection/recovery step.")
    parser.add_argument('--outdir_suffix', type=str, default='', help="Suffix to add to the end of the output directory where results will be saved.")
    parser.add_argument('--use_cps_date_grid', action='store_true', help="If used, then use the CPS date grid for generating observations. Note, may break if obs cadence and Nobs are incompatible.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--add_red_noise', action='store_true', help='If used, then add red noise in the form of a GP prediction at the observation dates. NOTE: Minimum working implementation right now.')
    args = parser.parse_args()
    return args

def main():
    # Parse the command line arguments
    args = parse_args()

    # Set the random seed
    np.random.seed(args.seed)

    # Make the output directory
    outdir = f"{args.mpsini:02}_{args.period:02}_{args.rverr_internal:03.1f}_{args.rverr_astro:03.1f}"
    if args.outdir_suffix != '':
        outdir += f'_{args.outdir_suffix}'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Generate the synthetic RV time series
    star = RVStar()
    star.set_mstar(args.mstar)
    k_truth = radvel.utils.semi_amplitude(args.mpsini, args.period, star.get_mstar(), e, Msini_units='earth')
    tp = np.random.uniform(0, args.period)
    orbel = (args.period, tp, e, omega, k_truth)
    planet = RVPlanet(orbel=orbel)
    star.add_planet(planet)
    
    sim = SynthSim(star, rv_meas_err=args.rverr_internal, astro_jitter=args.rverr_astro, obs_bjd_start=obs_bjd_start)
    sim.set_random_seed(args.seed)
    sim.set_nobs(math.ceil(args.nobs * (1 + args.weather_losses)))
    sim.set_min_obs_cadence(int(planet.per / args.obs_cadence))

    time_grid = sim.get_obs_dates(use_cps_date_grid=args.use_cps_date_grid)
    time_grid += np.random.uniform(-args.time_jit, args.time_jit, size=len(time_grid))
    keep_inds = np.random.choice(np.arange(len(time_grid)), size=args.nobs, replace=False)
    time_grid = np.sort(time_grid[keep_inds]) # Create some gaps in the observations due to weather losses.
    sim.set_nobs(len(time_grid))
    df_rv = sim.get_simulated_obs(time_grid)
    df_rv.to_csv(os.path.join(outdir, 'synthetic_rvs.csv'), index=False)
    plot_rvs(sim, df_rv, args.mpsini, args.period, outdir)

    # Conduct an initial fit of the RVs using RadVel to get a baseline for the K-amplitude of the known planet
    tc = radvel.orbit.timeperi_to_timetrans(orbel[1], orbel[0], e, omega, secondary=False)
    priors = [radvel.prior.Gaussian('per1', args.period, 1e-3), radvel.prior.Gaussian('tc1', tc, 1e-1)]
    post = sim.get_radvel_post(priors=priors)
    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    k1_map_baseline = post.params['k1'].value # m/s

    # Run MCMC to estimate uncertainties on k1
    df_mcmc = radvel.mcmc(post, save=True, savename=os.path.join(outdir, 'baseline_chains.h5'))
    k1_med_baseline = np.median(df_mcmc['k1'])
    k1_err_baseline = np.mean(np.abs(np.quantile(df_mcmc['k1'], [0.16, 0.86]) - k1_med_baseline))

    # Plot the results
    post.medparams = {'per1':args.period, 'k1':round(k1_med_baseline, 2), 'e1':e}
    radvel_plot = MultipanelPlot(post, uparams={'per1':0, 'k1':round(k1_err_baseline, 2), 'e1':0})
    fig, axes = radvel_plot.plot_multipanel()
    fig.savefig(os.path.join(outdir, 'baseline_radvel_fit.png'), dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()

    # If specified, add red noise
    if args.add_red_noise:

        # Picked sort of arbitrarily but sort of in line with the results for the Sun from Kosiarek & Crossfield (2020)
        eta1 = 2 # m/s
        eta2 = 35 # days
        eta3 = 27 # days
        eta4 = 0.5
        solar_qp_hyperparams = (eta1, eta2, eta3, eta4)
        red_noise = get_red_noise(time_grid, qp_kernel, solar_qp_hyperparams)
        df_rv['mnvel'] += red_noise
        sim.data = df_rv # Important step
        df_rv.to_csv(os.path.join(outdir, 'synthetic_rvs_w_red_noise.csv'), index=False)
        plot_rvs(sim, sim.data, args.mpsini, args.period, outdir, fname_suffix='_w_red_noise', red_noise_arr=red_noise)

        # Conduct an initial fit of the RVs that include the red noise but don't model the red noise to get a
        # sense of how it messes with things before adding additional planetary signals on top. 
        post = sim.get_radvel_post(priors=priors)
        post = radvel.fitting.maxlike_fitting(post, verbose=True)
        # Plot the results
        radvel_plot = MultipanelPlot(post)
        fig, axes = radvel_plot.plot_multipanel()
        fig.savefig(os.path.join(outdir, 'keplerian_radvel_fit_w_red_noise.png'), dpi=200, facecolor='white', bbox_inches='tight')
        plt.close()
    
    if args.recoveries_fname is None:
        # Run RVSearch on the synthetic RVs so we can then do injection recovery tests
        search_obj = rvsearch.Search(df_rv, post=post, mcmc=False, min_per=plim[0], max_per=plim[1]) # What to do if RVSearch says there is more than one signal? Maybe doesn't matter for now
        try:
            search_obj.run_search(outdir=outdir)
        except ValueError: # RVSearch runs into a RadVel plotting error if there are no planets detected 
            print('Original injected planet not detected in RV time series.')
            pass

        # Run the injection and recoveries
        pickle_fname = os.path.join(outdir, 'search.pkl')
        recoveries_obj = rvsearch.inject.Injections(pickle_fname, plim, klim, (0.0, 0.0), num_sim=args.num_sim, seed=args.seed)
        df_recover = recoveries_obj.run_injections(num_cpus=args.num_cpus)
        df_recover_fname = os.path.join(outdir, 'recoveries.csv')
        df_recover.to_csv(df_recover_fname, index=False)

        # Calculate the completeness map
        comp = rvsearch.inject.Completeness(df_recover, 'inj_period', 'inj_k', mstar=args.mstar)

        # Plot the completeness map
        xi, yi, zi = comp.completeness_grid(xlim=plim, ylim=klim, resolution=30)
        cp = rvsearch.plots.CompletenessPlots(comp, searches=[search_obj])
        fig = cp.completeness_plot(xlabel='$P$ [d]', ylabel='$K$ [m/s]')
        fig.savefig(os.path.join(outdir, 'completeness.png'), dpi=200, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        print(f"Loading recoveries file from {args.recoveries_fname}")
        df_recover_fname = args.recoveries_fname
        df_recover = pd.read_csv(df_recover_fname)

    # Choose a random sample of failed recoveries from RVSearch
    num_recoveries = np.sum(df_recover.recovered == False)
    n_recover_test = 1000 # Number of RVSearch recoveries to test by adding to the synthetic time series and then re-fitting
    if n_recover_test > num_recoveries:
        n_recover_test = num_recoveries
    recover_test_inds = np.random.choice(df_recover[df_recover.recovered == False].index, size=n_recover_test, replace=False)

    df_recover_test = pd.DataFrame(columns=['inj_period', 'inj_tp', 'inj_k', 'k1'])

    ind = recover_test_inds[0]
    orbel = (df_recover.loc[ind, 'inj_period'], df_recover.loc[ind, 'inj_tp'], 0, np.pi/2, df_recover.loc[ind, 'inj_k'])
    planet = RVPlanet(pl_letter='c', orbel=orbel)
    sim.star.add_planet(planet)
    df = sim.get_simulated_obs(time_grid) # Add the planet to the time series data, but then get rid of it so that RadVel only fits a 1-planet model
    sim.star.pop_planet('c')
    
    # For some reason RVSearch has the ability to change the local definition of the "priors" variable after it runs, so adding this line to reset things.
    # Fixes issue with KeyError for a second planet.
    priors = [radvel.prior.Gaussian('per1', args.period, 1e-3), radvel.prior.Gaussian('tc1', tc, 1e-1)]
    post = sim.get_radvel_post(priors=priors)
    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    df_recover_test.loc[0, 'inj_period'] = orbel[0]
    df_recover_test.loc[0, 'inj_tp'] = orbel[1]
    df_recover_test.loc[0, 'inj_k'] = orbel[-1]
    df_recover_test.loc[0, 'k1'] = post.params['k1'].value

    j = 1
    for ind in tqdm(recover_test_inds[1:]):
        orbel = (df_recover.loc[ind, 'inj_period'], df_recover.loc[ind, 'inj_tp'], 0, np.pi/2, df_recover.loc[ind, 'inj_k'])
        planet = RVPlanet(pl_letter='c', orbel=orbel)
        sim.star.add_planet(planet)
        df = sim.get_simulated_obs(time_grid) # Add the planet to the time series data, but then get rid of it so that RadVel only fits a 1-planet model
        sim.star.pop_planet('c')
        post = sim.get_radvel_post(priors=priors)
        post = radvel.fitting.maxlike_fitting(post, verbose=False)
        df_recover_test.loc[j, 'inj_period'] = orbel[0]
        df_recover_test.loc[j, 'inj_tp'] = orbel[1]
        df_recover_test.loc[j, 'inj_k'] = orbel[-1]
        df_recover_test.loc[j, 'k1'] = post.params['k1'].value

        # Plot the recovery tests
        recovery_test_plotting_outdir = os.path.join(outdir, "recovery_test_plotting")
        if not os.path.isdir(recovery_test_plotting_outdir):
            os.mkdir(recovery_test_plotting_outdir) # Make the output directory
        if args.add_red_noise:
            plot_rvs(sim, df, args.mpsini, args.period, recovery_test_plotting_outdir, fname_suffix=f'_inj_period_{orbel[0]:.2f}_injk_{orbel[-1]:.2f}', red_noise_arr=red_noise)
        else:
            plot_rvs(sim, df, args.mpsini, args.period, recovery_test_plotting_outdir, fname_suffix=f'_inj_period_{orbel[0]:.2f}_injk_{orbel[-1]:.2f}')
        radvel_plot = MultipanelPlot(post)
        fig, axes = radvel_plot.plot_multipanel()
        fig.savefig(os.path.join(recovery_test_plotting_outdir, f'radvel_fit_inj_period_{orbel[0]:.2f}_injk_{orbel[-1]:.2f}.png'), dpi=200, facecolor='white', bbox_inches='tight')
        plt.close()

        j += 1

    df_recover_test['k1_fit_over_k1_truth'] = df_recover_test['k1'] / k_truth
    df_recover_test['k1_fit_over_k1_map_baseline'] = df_recover_test['k1'] / k1_map_baseline
    df_recover_test['k1_fit_over_k1_med_baseline'] = df_recover_test['k1'] / k1_med_baseline
    df_recover_test['k1_fit_minus_k1_med_baseline_over_k1_err_baseline'] = (df_recover_test['k1'] - k1_med_baseline) / k1_err_baseline
    df_recover_test.to_csv(os.path.join(outdir, 'recover_test_results.csv'), index=False)

    # Plot the results of running the recoveries test
    fig, ax = plot_recovery_test_results(df_recover_fname, df_recover_test, outdir, mstar=args.mstar, plim=plim, klim=klim)

if __name__ == '__main__':
    main()