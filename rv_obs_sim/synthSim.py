import numpy as np
from radvel.kepler import rv_drive

class RVPlanet:

    def __init__(self, pl_letter:str='b', orbel:tuple=None) -> None:
        self.pl_letter = pl_letter
        self.orbel = orbel
        if self.orbel is not None:
            self.p, self.tp, self.e, self.omega, self.k = self.orbel
    
    def __str__(self) -> str:
        str_out  = '-----'
        str_out += f"Planet {self.pl_letter}\n"
        str_out += f"P = {self.p:10} d\n"
        str_out += f"Tperi = {self.tp:10} BJD\n"
        str_out += f"Ecc = {self.e:10}\n"
        str_out += f"Omega = {self.omega:10}\n"
        str_out += f"K = {self.k:10} m/s\n"
        str_out += '-----'
        return str_out
        
    def set_orbel(self, orbel:tuple, bjd_offset:float=0) -> None:
        self.orbel = orbel
        self.p, self.tp, self.e, self.omega, self.k = self.orbel
        self.bjd_offset = self.bjd_offset

class RVStar:

    def __init__(self, sys_name:str='') -> None:
        self.sys_name = sys_name
        self.mstar = None
        self.planets = {}

    def __str__(self) -> str:
        str_out = '-----'
        str_out += f"Star: {self.sys_name}\n"
        str_out += f"Mstar: {self.mstar}\n"
        str_out += f"-----"
        for planet in self.planets.values():
            str_out += planet.__str__()

    def add_planet(self, rvplanet:RVPlanet) -> None:
        self.planets[rvplanet.pl_letter] = rvplanet

        planet_b_bjd_offset = self.planets['b'].bjd_offset
        assert [planet.bjd_offset == planet_b_bjd_offset for planet in self.planets.values()], 'All planets must have same bjd offset for Tc and/or Tperi values.'

    def pop_planet(self, pl_letter:str) -> RVPlanet:
        return self.planets.pop(pl_letter)

    def set_mstar(self, mstar:float) -> None:
        self.mstar = mstar
    
    def get_mstar(self) -> float:
        return self.mstar

class RVSystem(RVStar):
    
    def __init__(self, rvstar:RVStar) -> None:
        self.rvstar = rvstar
        self.rvplanets = self.rvstar.planets

class RVObsSim(RVSystem):
    '''
    Simulate synthetic RV observations. 

    Can only use one telescope for now.
    '''

    def __init__(self, rvsystem:RVSystem, tel_name:str='HIRES', rv_meas_err:float=2, tel_jitter:float=0, astro_jitter:float=0) -> None:
        
        self.rvsystem = rvsystem
        self.tel_name = tel_name
        self.rv_meas_err = rv_meas_err
        self.tel_jitter = tel_jitter
        self.astro_jitter = astro_jitter

        # Assumes that bjd_offset is the same for each planet
        self.bjd_offset = self.rvsystem.rvplanets['b'].bjd_offset

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
            if self.obs_cadence * self.nobs > self.N_DATE_GRID:
                self.N_DATE_GRID = round((self.obs_cadence * self.nobs) * 10, -3) # Round to nearest 1000
        except:
            pass

    def set_obs_cadence(self, obs_cadence:int) -> None:
        '''
        Set the observing cadence goal in units of Number of calendar nights between observations.
        e.g.,
        obs_cadence =  1 -> Nightly observations
        obs_cadence = 10 -> 1 observation every 10 days
        '''
        self.obs_cadence = obs_cadence
        try:
            if self.obs_cadence * self.nobs > self.N_DATE_GRID:
                self.N_DATE_GRID = round((self.obs_cadence * self.nobs) * 10, -3) # Round to nearest 1000
        except:
            pass

    def get_obs_dates(self) -> object:
        '''
        Pick which dates the observations will be taken
        '''
        date_grid = np.arange(self.N_DATE_GRID) + self.bjd_offset
        obs_dates = date_grid[::self.obs_cadence]
        
        # This is where you would apply masks for: (1) observability, (2) bad weather, (3) telescope schedule, etc.
        mask = np.ones(len(obs_dates), dtype=bool)
        
        obs_dates = obs_dates[mask][:self.nobs]
        self.obs_dates = obs_dates
        return self.obs_dates

    def get_simulated_obs(self) -> tuple:
        
        rv_tot = np.zeros(len(self.obs_dates))

        for planet in self.rvsystem.rvplanets.values():
            rv_tot += rv_drive(self.obs_dates, planet.orbel, use_c_kepler_solver=False) # C solver not working for some reason
        
        rv_tot += np.random.normal(scale=np.sqrt(self.astro_jitter**2 + self.tel_jitter**2))
        
        return (self.obs_dates, rv_tot, self.rv_meas_err, self.tel_name)
    

    