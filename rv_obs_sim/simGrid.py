import numpy as np
import radvel
from tqdm import tqdm

class SynthSimGrid:
    
    def __init__(self, moc_grid, nrv_grid, base_config_file, fit_config_file, astro_jitter=0, tel_jitter=0, errvel_scale=2, max_baseline=3650) -> None:

        self.moc_grid = moc_grid
        self.nrv_grid = nrv_grid

        self.base_config_file = base_config_file
        self.fit_config_file = fit_config_file

        self.astro_jitter = astro_jitter
        self.tel_jitter = tel_jitter
        self.errvel_scale = errvel_scale

        self.max_baseline = max_baseline

    def __fit_and_get_k_maps(self, post, mask, planet_letter_keys, verbose=False):
        
        post.likelihood.x = self.time_grid[mask]
        post.likelihood.y = self.mnvel_grid[mask]
        post.likelihood.yerr = self.errvel_grid[mask]

        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
        return np.array([post.params[f'k{i}'].value for i in planet_letter_keys])

    def __set_parent_synth_data_grid(self):
        '''
        Should only have to generate the synthetic RVs once, but on a large grid, and then you can just 
        resample them later based on the number of RVs requested and the MOC.
        '''
        base_config_file_obj, base_post = radvel.utils.initialize_posterior(self.base_config_file)
        time_grid = np.arange(self.max_baseline) + base_config_file_obj.time_base

        rv_tot = np.zeros(len(time_grid))

        for pl_ind in base_config_file_obj.planet_letters.keys(): # Need to input correct K-amplitudes in config file.
            
            # Extract orbit parameters
            p = base_post.params[f'per{pl_ind}'].value
            tc = base_post.params[f'tc{pl_ind}'].value
            ecc = base_post.params[f'ecc{pl_ind}'].value
            omega = base_post.params[f'omega{pl_ind}'].value
            tp = radvel.orbit.timetrans_to_timeperi(tc, p, ecc, omega)
            k = base_post.params[f'k{pl_ind}'].value
            
            orbel = (p, tp, ecc, omega, k)
            rv_tot += radvel.kepler.rv_drive(time_grid, orbel)

        # Add background trend if applicable. Need to input correct dvdt and curv values in config file.
        rv_tot += (time_grid - base_config_file_obj.time_base) * base_post.params['dvdt'].value
        rv_tot += (time_grid - base_config_file_obj.time_base)**2 * base_post.params['curv'].value

        # Add random noise
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_jitter**2 + self.tel_jitter**2), size=len(rv_tot))

        self.base_post = base_post
        self.base_config_file_obj = base_config_file_obj

        self.time_grid, self.mnvel_grid, self.errvel_grid = time_grid, rv_tot, self.errvel_scale * np.ones(len(rv_tot))
        
    def get_ksim_over_ktruth_grid(self, disable_progress_bar=False):

        self.__set_parent_synth_data_grid()
        fit_config_file_obj, fit_post = radvel.utils.initialize_posterior(self.fit_config_file) 

        ksim_over_ktruth = np.ones((len(self.moc_grid), len(self.nrv_grid), fit_post.model.num_planets))
        for k in fit_config_file_obj.planet_letters.keys():
            ksim_over_ktruth[:, :, k - 1] /= self.base_post.params[f'k{k}'].value
        
        i = 0
        for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
            for j, nrv in enumerate(self.nrv_grid):
                mask = (self.time_grid - self.config_file_obj.time_base) % moc == 0
                mask &= (self.time_grid - self.config_file_obj.time_base) / moc < nrv
                k_maps = self.__fit_and_get_k_maps(fit_post, mask, list(fit_config_file_obj.planet_letters.keys()), verbose=False)

                for k in range(fit_post.model.num_planets):
                    ksim_over_ktruth[i, j, k] *= k_maps[k]

            i += 1
        
        return ksim_over_ktruth