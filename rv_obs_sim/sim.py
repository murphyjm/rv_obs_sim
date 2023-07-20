import numpy as np
import pandas as pd
from radvel.kepler import rv_drive

class RVStar:

    def __init__(self, sys_name:str='') -> None:

        self.sys_name = sys_name

class RVPlanet(RVStar):

    def __init__(self, rvstar, pl_letter:str='b') -> None:

        self.star = rvstar
        self.pl_letter = pl_letter

class RVSystem:
    
    def __init__(self, rvstar:RVStar, rvplanet_list:RVPlanet) -> None:
        
        self.rvstar = rvstar
        

class RVObsSim(RVSystem):

    def __init__(self, rvsystem:RVSystem) -> None:
        self.rvsystem = rvsystem

    def set_nobs(self, nobs:int) -> None:
        self.nobs = nobs

    def set_obs_cadence(self, obs_cadence:int) -> None:
        '''
        Set the observing cadence goal in units of Number of calendar nights between observations.
        e.g.,
        obs_cadence =  1 -> Nightly observations
        obs_cadence = 10 -> 1 observation every 10 days
        '''
        self.obs_cadence = obs_cadence

    def load_data(self, data_path, read_csv_kwargs):
        
        try:
            df = pd.read_csv(data_path, **read_csv_kwargs)
        except FileNotFoundError:
            print(f"Warning: Data path {data_path} does not exist.")
            return

        assert set(['time', 'mnvel','errvel', 'tel']).issubset(df.columns), "Data must have columns: 'time', 'mnvel', 'errvel', 'tel'."

        self.time = df.time.values
        self.mnvel_obs = df.mnvel.values
        self.errvel_obs = df.errvel.values
        self.tel = df.tel.values

    def add_planet(self, planet):
        pass