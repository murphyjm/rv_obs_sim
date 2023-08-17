import numpy as np
import pandas as pd
import radvel
from tqdm import tqdm

class SimGrid:
    
    def __init__(self, moc_grid, nrv_grid, base_config_file, fit_config_file, 
                 data_file=None, 
                 obs_start_end=(None, None),
                 read_csv_kwargs={},
                 random_seed=42,
                 tel='hires_j', 
                 time_jitter=0, astro_rv_jitter=0, tel_rv_jitter=0, errvel_scale=2, 
                 max_baseline=3650) -> None:

        self.moc_grid = moc_grid
        self.nrv_grid = nrv_grid

        self.base_config_file = base_config_file
        base_config_file_obj, base_post = radvel.utils.initialize_posterior(self.base_config_file)
        self.base_config_file_obj = base_config_file_obj
        self.base_post = base_post

        self.fit_config_file = fit_config_file
        self.data_file = data_file
        self.obs_start, self.obs_end = obs_start_end
        if self.data_file is not None:
            self.__load_data(**read_csv_kwargs)

        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.tel = tel
        self.time_jitter = time_jitter
        self.astro_rv_jitter = astro_rv_jitter
        self.tel_rv_jitter = tel_rv_jitter
        self.errvel_scale = errvel_scale

        self.max_baseline = max_baseline

    def __load_data(self,  **read_csv_kwargs):

        try:
            df = pd.read_csv(self.data_file, **read_csv_kwargs)
        except FileNotFoundError:
            print(f"Warning: Data path {self.data_file} does not exist.")
            return

        if 'bjd' in df.columns.tolist():
            df = df.rename(columns={'time':'date', 'bjd':'time'})

        assert set(['time', 'mnvel','errvel', 'tel']).issubset(df.columns), "Data must have columns: 'time', 'mnvel', 'errvel', 'tel'."
        
        data = pd.DataFrame()

        bintime, binmnvel, binerrvel, bintel = radvel.utils.bintels(df.time, df.mnvel, df.errvel, df.tel, binsize=0.1)
        data['time'] = bintime
        data['mnvel'] = binmnvel
        data['errvel'] = binerrvel
        data['tel'] = bintel
        self.data = data

        if self.obs_start is None:
            self.obs_start = np.min(self.data.time) - 1
        if self.obs_end is None:
            self.obs_end = np.max(self.data.time) + 1

    def __set_parent_synth_data_grid(self):
        '''
        Should only have to generate the synthetic RVs once, but on a large grid, and then you can just 
        resample them later based on the number of RVs requested and the MOC.
        '''

        time_grid = (np.arange(self.max_baseline) + self.base_config_file_obj.time_base).astype(float)
        time_grid += np.random.normal(scale=self.time_jitter, size=len(time_grid))

        rv_tot = np.zeros(len(time_grid))

        for pl_ind in self.base_config_file_obj.planet_letters.keys(): # Need to input correct K-amplitudes in config file.
            
            # Extract orbit parameters
            p = self.base_post.params[f'per{pl_ind}'].value
            tc = self.base_post.params[f'tc{pl_ind}'].value
            if 'secosw' in self.base_config_file_obj.fitting_basis and 'sesinw' in self.base_config_file_obj.fitting_basis:
                secosw = self.base_post.params[f'secosw{pl_ind}'].value
                sesinw = self.base_post.params[f'secosw{pl_ind}'].value
                e = secosw**2 + sesinw**2
                omega = np.arctan(sesinw / secosw)
                if np.isnan(omega):
                    omega = 0.0
            else:
                e = self.base_post.params[f'e{pl_ind}'].value
                omega = self.base_post.params[f'omega{pl_ind}'].value
            tp = radvel.orbit.timetrans_to_timeperi(tc, p, e, omega)
            k = self.base_post.params[f'k{pl_ind}'].value
            
            orbel = (p, tp, e, omega, k)
            rv_tot += radvel.kepler.rv_drive(time_grid, orbel)

        # Add background trend if applicable. Need to input correct dvdt and curv values in config file.
        rv_tot += (time_grid - self.base_config_file_obj.time_base) * self.base_post.params['dvdt'].value
        rv_tot += (time_grid - self.base_config_file_obj.time_base)**2 * self.base_post.params['curv'].value

        # Add random noise
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_rv_jitter**2 + self.tel_rv_jitter**2), size=len(rv_tot))

        self.time_grid, self.mnvel_grid, self.errvel_grid = time_grid, rv_tot, self.errvel_scale * np.ones(len(rv_tot))

    def __fit_and_get_post(self, post, verbose=False):
        post.priors += [radvel.prior.HardBounds(f'jit_{self.tel}', 0.0, 20.0)] # HACK! Make sure this matches the priors in fit_config_file.py!!
        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
        return post

    def get_ksim_over_ktruth_grid(self, disable_progress_bar=False, save_posts=False):

        if self.data_file is None:
            self.__set_parent_synth_data_grid()

        fit_config_file_obj, fit_post = radvel.utils.initialize_posterior(self.fit_config_file)

        if save_posts:
            post_grid = np.empty((len(self.moc_grid), len(self.nrv_grid)), dtype=object)

        ksim_over_ktruth = np.ones((len(self.moc_grid), len(self.nrv_grid), fit_post.model.num_planets))
        for k in fit_config_file_obj.planet_letters.keys():
            ksim_over_ktruth[:, :, k - 1] /= self.base_post.params[f'k{k}'].value
        
        if self.data_file is None:
            i = 0
            time_grid_inds = np.arange(self.max_baseline)
            for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
                moc_mask = time_grid_inds % moc == 0

                for j, nrv in enumerate(self.nrv_grid):

                    # Note: fit_config_file_obj.time_base and self.base_config_file_obj.time_base should probably be the same.
                    mask = moc_mask & (time_grid_inds / moc < nrv)

                    fit_mod = radvel.RVModel(fit_post.params, time_base=fit_config_file_obj.time_base)
                    fit_like = radvel.likelihood.RVLikelihood(fit_mod, self.time_grid[mask], self.mnvel_grid[mask], self.errvel_grid[mask])
                    post = radvel.posterior.Posterior(fit_like)
                    post = self.__fit_and_get_post(post, verbose=False)
                    planet_letter_keys = list(fit_config_file_obj.planet_letters.keys())
                    k_maps = np.array([post.params[f'k{i}'].value for i in planet_letter_keys])

                    for k in range(fit_post.model.num_planets):
                        ksim_over_ktruth[i, j, k] *= k_maps[k]

                    if save_posts:
                        post_grid[i, j] = post

                i += 1
        else:
            time_range_mask = self.data['time'] > self.obs_start
            time_range_mask &= self.data['time'] < self.obs_end
            inds = self.data[time_range_mask].index
            
            i = 0
            for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
                for j, nrv in enumerate(self.nrv_grid):
                    good_inds = [inds[0]]
                    prev_time = self.data.loc[good_inds[0], 'time']
                    for k in range(1, len(inds)):
                        if len(good_inds) >= nrv:
                            break
                        ind = inds[k]
                        if (self.data.loc[ind, 'time'] - prev_time) >= moc:
                            good_inds.append(ind)
                            prev_time = self.data.loc[ind, 'time']
                        else:
                            continue
                    
                    resampled_data = self.data.iloc[good_inds]
                    fit_mod = radvel.RVModel(fit_post.params, time_base=fit_config_file_obj.time_base)
                    fit_like = radvel.likelihood.RVLikelihood(fit_mod, resampled_data.time, resampled_data.mnvel, resampled_data.errvel)
                    post = radvel.posterior.Posterior(fit_like)
                    post = self.__fit_and_get_post(post, verbose=False)
                    planet_letter_keys = list(fit_config_file_obj.planet_letters.keys())
                    k_maps = np.array([post.params[f'k{i}'].value for i in planet_letter_keys])

                    for k in range(fit_post.model.num_planets):
                        ksim_over_ktruth[i, j, k] *= k_maps[k]

                    if save_posts:
                        post_grid[i, j] = post

                i += 1
        
        self.ksim_over_ktruth = ksim_over_ktruth
        if save_posts:
            self.post_grid = post_grid
            
        return ksim_over_ktruth