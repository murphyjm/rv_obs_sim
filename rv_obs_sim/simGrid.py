import numpy as np
import radvel
from tqdm import tqdm


class SynthSimGrid:
    
    def __init__(self, nrv_grid, moc_grid, config_file, bjd_offset:float=0) -> None:

        self.nrv_grid = nrv_grid
        self.moc_grid = moc_grid
        self.config_file = config_file
        self.bjd_offset = bjd_offset

        self.__N_DATE_GRID = 1000 # Some arbitrarily large number, but should be larger than max(self.moc_grid) * max(self.nrv_grid) > self.N_DATE_GRID:
        if np.max(self.moc_grid) * np.max(self.nrv_grid) > self.__N_DATE_GRID:
            self.__N_DATE_GRID = round((np.max(self.moc_grid) * np.max(self.nrv_grid)) * 10, -2) # Round to nearest 100

        self.__parent_synth_data_grid = self.__get_parent_synth_data_grid(self)

    def __fit_and_get_k_maps(self, config_file, verbose=False):
        P, post = radvel.utils.initialize_posterior(config_file)
       
        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)

        return [post.likelihood.params[f'k{i}'].value for i in range(1, post.model.num_planets + 1)]

    def __get_parent_synth_data_grid(self):
        '''
        Should only have to generate the synthetic RVs once, but on a large grid, and then you can just 
        resample them later based on the number of RVs requested and the MOC.
        '''
        P, post = radvel.utils.initialize_posterior(self.config_file)
        date_grid = np.arange(self.__N_DATE_GRID) + self.obs_bjd_start

        rv_tot = np.zeros(len(date_grid))

        for planet in self.star.planets.values(): # Need to input correct K-amplitudes here
            rv_tot += radvel.kepler.rv_drive(self.obs_dates, planet.orbel)
        
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_jitter**2 + self.tel_jitter**2), size=len(rv_tot))

        rv_tot += self.obs_dates * self.star.dvdt

        
    def generate_fit_grid(self, disable_progress_bar=False):

        for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
            for nrv in self.nrv_grid:
                pass
                