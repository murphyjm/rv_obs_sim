import numpy as np
import pandas as pd

class DataSim:

    def __init__(self, nobs, min_obs_cadence, obs_start_end=(None, None)) -> None:
        '''
        Minimum obs cadence is in units of Number of calendar nights between observations.
        e.g., min_obs_cadence =  1 -> Take observations every night, but no more frequently than that. obs_cadence = 10 -> 1 observation every >=10 days. etc.
        
        ***For now, only trust this object when using a dataset from a single telescope***
        '''
        assert (nobs > 0 and min_obs_cadence > 0), 'nobs and obs_cadence must both be > 0.'
        self.nobs = nobs
        self.min_obs_cadence = min_obs_cadence
        self.obs_start, self.obs_end = obs_start_end

        self.data = None
    
    def load_data(self, data_path, read_csv_kwargs) -> None:
    
        try:
            df = pd.read_csv(data_path, **read_csv_kwargs)
        except FileNotFoundError:
            print(f"Warning: Data path {data_path} does not exist.")
            return

        assert set(['time', 'mnvel','errvel', 'tel']).issubset(df.columns), "Data must have columns: 'time', 'mnvel', 'errvel', 'tel'."
        self.data = df

        if self.obs_start is None:
            self.obs_start = np.min(self.data.time) - 1
        if self.obs_end is None:
            self.obs_end = np.max(self.data.time) + 1

    def resample_data(self) -> pd.DataFrame:
        '''
        Resample the data according to the specified number of observations, cadence, and start/end dates.
        '''
        df = pd.DataFrame(columns=['time', 'mnvel', 'errvel', 'tel'])
        inds = -1 * np.ones(self.nobs) # Try to meet the number of observations requested, but may not be possible.

        mask = self.data['time'] > self.obs_start
        mask &= self.data['time'] < self.obs_end
        mask &= np.array([1] + [np.ediff1d(self.data['time']) >= self.min_obs_cadence], dtype=bool)
        mask &= np.arange(len(self.data)) < self.nobs

        self.df_resampled = df[mask]
        return self.df_resampled