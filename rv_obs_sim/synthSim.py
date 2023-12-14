import numpy as np
import pandas as pd
import radvel

class RVPlanet:

    def __init__(self, pl_letter:str='b', orbel:tuple=None) -> None:
        self.pl_letter = pl_letter
        self.orbel = orbel
        if self.orbel is not None:
            self.set_orbel(self.orbel)
    
    def __str__(self) -> str:
        str_out  = '-----'
        str_out += f"Planet {self.pl_letter}\n"
        str_out += f"P = {self.per:10} d\n"
        str_out += f"Tperi = {self.tp:10} BJD\n"
        str_out += f"Ecc = {self.e:10}\n"
        str_out += f"Omega = {self.omega:10}\n"
        str_out += f"K = {self.k:10} m/s\n"
        str_out += '-----'
        return str_out
        
    def set_orbel(self, orbel:tuple) -> None:
        '''
        Note: Tp is time of periastron passage **not** time of transit.
        '''
        self.orbel = orbel
        self.per, self.tp, self.e, self.omega, self.k = self.orbel

class RVStar:

    def __init__(self, sys_name:str='') -> None:
        self.sys_name = sys_name
        self.mstar = None
        self.planets = {}
        self.dvdt = 0

    def __str__(self) -> str:
        str_out = '-----'
        str_out += f"Star: {self.sys_name}\n"
        str_out += f"Mstar: {self.mstar}\n"
        str_out += f"-----"
        for planet in self.planets.values():
            str_out += planet.__str__()

    def add_planet(self, rvplanet:RVPlanet) -> None:
        self.planets[rvplanet.pl_letter] = rvplanet

    def pop_planet(self, pl_letter:str) -> RVPlanet:
        return self.planets.pop(pl_letter)

    def set_mstar(self, mstar:float) -> None:
        self.mstar = mstar
    
    def get_mstar(self) -> float:
        return self.mstar
    
    def set_dvdt(self, dvdt:float):
        self.dvdt = dvdt

class SynthSim(RVStar):
    '''
    Simulate synthetic RV observations. 

    Can only use one telescope for now.
    '''

    def __init__(self, star:RVStar, 
                 tel_name:str='hires_j', 
                 rv_meas_err:float=2, 
                 tel_jitter:float=0, 
                 astro_jitter:float=0,
                 obs_bjd_start:float=2457000) -> None:
        self.star = star
        self.tel_name = tel_name
        self.rv_meas_err = rv_meas_err
        self.tel_jitter = tel_jitter
        self.astro_jitter = astro_jitter

        self.obs_bjd_start = obs_bjd_start

        self.seed = 42
        np.random.seed(self.seed)
        
        # A helper variable for setting the grid from which to draw observation times
        self.N_DATE_GRID = 1000

    def set_random_seed(self, seed:int) -> None:
        self.seed = seed
        np.random.seed(self.seed)

    def set_nobs(self, nobs:int) -> None:
        self.nobs = nobs
        try:
            if self.min_obs_cadence * self.nobs > self.N_DATE_GRID:
                self.N_DATE_GRID = round((self.min_obs_cadence * self.nobs) * 10, -3) # Round to nearest 1000
        except:
            pass

    def set_min_obs_cadence(self, min_obs_cadence:int) -> None:
        '''
        Set the observing cadence goal in units of Number of calendar nights between observations.
        e.g.,
        min_obs_cadence =  1 -> Nightly observations
        min_obs_cadence = 10 -> 1 observation every 10 days
        '''
        self.min_obs_cadence = min_obs_cadence
        try:
            if self.min_obs_cadence * self.nobs > self.N_DATE_GRID:
                self.N_DATE_GRID = round((self.min_obs_cadence * self.nobs) * 10, -3) # Round to nearest 1000
        except:
            pass

    def get_obs_dates(self, use_cps_date_grid=False) -> object:
        '''
        Pick which dates the observations will be taken
        '''
        if use_cps_date_grid:
            date_grid = pd.read_csv('cps_obs_dates_2023AB.csv')
            date_grid = date_grid.drop_duplicates().reset_index(drop=True)
            date_grid['time'] = pd.DatetimeIndex(date_grid['date']).to_julian_date()
            date_grid = date_grid.sort_values(by='time').reset_index(drop=True)
            date_grid = date_grid['time'].values
            if self.min_obs_cadence == 0:
                obs_dates = date_grid
            else:
                obs_dates = [date_grid[0]]
                for i in range(1, len(date_grid)):
                    if date_grid[i] - obs_dates[-1] < self.min_obs_cadence:
                        continue
                    else:
                        obs_dates.append(date_grid[i])
                obs_dates = np.asarray(obs_dates)
        else:
            date_grid = np.arange(self.N_DATE_GRID, dtype=float) + self.obs_bjd_start
            if self.min_obs_cadence == 0:
                obs_dates = date_grid
            else:
                obs_dates = date_grid[::self.min_obs_cadence]

        # This is where you would apply masks for: (1) observability, (2) bad weather, (3) telescope schedule, etc.
        mask = np.ones(len(obs_dates), dtype=bool)
        
        obs_dates = obs_dates[mask][:self.nobs]
        return obs_dates

    def get_simulated_obs(self, time_grid) -> tuple:
        
        rv_tot = np.zeros(len(time_grid))

        for planet in self.star.planets.values():
            rv_tot += radvel.kepler.rv_drive(time_grid, planet.orbel, use_c_kepler_solver=False) # C solver not working for some reason
        
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_jitter**2 + self.tel_jitter**2), size=len(rv_tot))

        rv_tot += time_grid * self.star.dvdt
        
        df = pd.DataFrame()
        df['time'], df['mnvel'], df['errvel'], df['tel'] = time_grid, rv_tot, self.rv_meas_err, self.tel_name
        self.data = df
        return df
    
    def get_radvel_post(self, priors=[]) -> radvel.posterior.Posterior:
        '''
        Take a simulation object and get a RadVel posterior object.
        '''
        nplanets = len(self.star.planets)
        fitting_basis = 'per tc secosw sesinw k'
        planet_letters = {num+1:letter for num, letter in zip(range(len(self.star.planets)), self.star.planets.keys())}

        # Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
        anybasis_params = radvel.Parameters(nplanets, basis='per tc e w k', 
                                            planet_letters=planet_letters) # initialize Parameters object
        for pl_num, pl_letter in planet_letters.items():
            per, tp, e, w, k = self.star.planets[pl_letter].orbel
            anybasis_params[f'per{pl_num}'] = radvel.Parameter(value=per)
            tc = radvel.orbit.timeperi_to_timetrans(tp, per, e, w, secondary=False)
            anybasis_params[f'tc{pl_num}'] = radvel.Parameter(value=tc)
            anybasis_params[f'e{pl_num}'] = radvel.Parameter(value=e)
            anybasis_params[f'w{pl_num}'] = radvel.Parameter(value=w)
            anybasis_params[f'k{pl_num}'] = radvel.Parameter(value=k)

        anybasis_params['dvdt'] = radvel.Parameter(value=self.star.dvdt)
        anybasis_params['curv'] = radvel.Parameter(value=0.0)

        bin_t, bin_vel, bin_err, bin_tel = radvel.utils.bintels(self.data['time'].values, self.data['mnvel'].values, self.data['errvel'].values, self.data['tel'].values, binsize=0.1)
        data = pd.DataFrame([], columns=['time', 'mnvel', 'errvel', 'tel'])
        data['time'] = bin_t
        data['mnvel'] = bin_vel
        data['errvel'] = bin_err
        data['tel'] = bin_tel

        time_base = np.median(self.data.time)
        instnames = np.unique(self.data.tel)
        for tel in instnames:
            anybasis_params[f'gamma_{tel}'] = radvel.Parameter(value=0.0)
            anybasis_params[f'jit_{tel}'] = radvel.Parameter(value=2.0)

        params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)

        iparams = radvel.basis._copy_params(params) # Taken from rvsearch.utils.initialize_post() but not sure why it's needed

        mod = radvel.RVModel(params, time_base=time_base)

        for i in range(nplanets):
            mod.params[f'per{i+1}'].vary = False
            mod.params[f'tc{i+1}'].vary = False
            mod.params[f'secosw{i+1}'].vary = False
            mod.params[f'sesinw{i+1}'].vary = False
        
        mod.params['dvdt'].vary = False
        mod.params['curv'].vary = False

        for tel in instnames:
            mod.params[f'gamma_{tel}'].vary = True
            mod.params[f'jit_{tel}'].vary = True

        inst_priors = []
        for tel in instnames:
            inst_priors.append(radvel.prior.HardBounds(f'gamma_{tel}', -100, 100))
            inst_priors.append(radvel.prior.HardBounds(f'jit_{tel}', 0, 20))

        priors += inst_priors
        
        # This block copied from rvsearch.utils.initialize_post()
        # >>>>>>>>>
        
        # initialize Likelihood objects for each instrument
        telgrps = self.data.groupby('tel').groups
        likes = {}

        for inst in telgrps.keys():
            # 10/8: ADD DECORRELATION VECTORS AND VARS, ONLY FOR SELECTED INST.
            likes[inst] = radvel.likelihood.RVLikelihood(mod, self.data.iloc[telgrps[inst]].time, self.data.iloc[telgrps[inst]].mnvel, self.data.iloc[telgrps[inst]].errvel, suffix='_'+inst)
            likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
            likes[inst].params['jit_'+inst] = iparams['jit_'+inst]
        # Can this be cleaner? like = radvel.likelihood.CompositeLikelihood(likes)
        like = radvel.likelihood.CompositeLikelihood(list(likes.values()))
        # <<<<<<<<<
        post = radvel.posterior.Posterior(like)
        post.priors = priors
        return post
