import numpy as np
import radvel
from tqdm import tqdm


class SynthSimGrid:
    
    def __init__(self, nrv_grid, moc_grid, config_file, bjd_offset:float=0, astro_jitter=0, tel_jitter=0) -> None:

        self.nrv_grid = nrv_grid
        self.moc_grid = moc_grid
        self.config_file = config_file
        self.bjd_offset = bjd_offset
        self.astro_jitter = astro_jitter
        self.tel_jitter = tel_jitter

        self.__N_DATE_GRID = 1000 # Some arbitrarily large number, but should be larger than max(self.moc_grid) * max(self.nrv_grid) > self.N_DATE_GRID:
        if np.max(self.moc_grid) * np.max(self.nrv_grid) > self.__N_DATE_GRID:
            self.__N_DATE_GRID = round((np.max(self.moc_grid) * np.max(self.nrv_grid)) * 10, -2) # Round to nearest 100

        self.__parent_synth_data_grid = self.__get_parent_synth_data_grid(self)

    def __fit_and_get_map_params(self, config_file, verbose=False):
        _, post = radvel.utils.initialize_posterior(config_file)
        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
        self.map_params = post.likelihood.params
        
        return post.likelihood.params

    def __get_parent_synth_data_grid(self):
        '''
        Should only have to generate the synthetic RVs once, but on a large grid, and then you can just 
        resample them later based on the number of RVs requested and the MOC.
        '''
        config_file_obj, init_post = radvel.utils.initialize_posterior(self.config_file)
        date_grid = np.arange(self.__N_DATE_GRID) + config_file_obj.time_base

        rv_tot = np.zeros(len(date_grid))

        for pl_ind in config_file_obj.planet_letters.keys(): # Need to input correct K-amplitudes in config file.
            
            # Extract orbit parameters
            p = init_post.params[f'per{pl_ind}'].value
            tc = init_post.params[f'tc{pl_ind}'].value
            ecc = init_post.params[f'ecc{pl_ind}'].value
            omega = init_post.params[f'omega{pl_ind}'].value
            tp = radvel.orbit.timetrans_to_timeperi(tc, p, ecc, omega)
            k = init_post.params[f'k{pl_ind}'].value
            
            orbel = (p, tp, ecc, omega, k)
            rv_tot += radvel.kepler.rv_drive(self.obs_dates, orbel)

        # Add background trend if applicable
        rv_tot += (self.obs_dates - config_file_obj.time_base) * config_file_obj.params['dvdt'].value
        rv_tot += (self.obs_dates - config_file_obj.time_base)**2 * config_file_obj.params['curv'].value

        # Add random noise
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_jitter**2 + self.tel_jitter**2), size=len(rv_tot))

        pass
        
    def generate_fit_grid(self, disable_progress_bar=False):

        for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
            for nrv in self.nrv_grid:
                pass
                