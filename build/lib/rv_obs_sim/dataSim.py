import numpy as np
import pandas as pd
from radvel.utils import bintels

class DataSim:

    def __init__(self, nobs, min_obs_cadence, obs_start_end=(None, None)) -> None:
        '''
        Minimum obs cadence is in units of Number of calendar nights between observations.
        e.g., min_obs_cadence =  1 -> Take observations every night, but no more frequently than that. obs_cadence = 10 -> 1 observation every >=10 days. etc.
        
        ***For now, only trust this object when using a dataset from a single telescope***
        '''
        assert (nobs >= 0 and min_obs_cadence >= 0), 'nobs and obs_cadence must both be >= 0.'
        self.nobs = nobs
        self.min_obs_cadence = min_obs_cadence
        self.obs_start, self.obs_end = obs_start_end

        self.data = None
    
    def load_data(self, data_path, **read_csv_kwargs) -> None:

        try:
            df = pd.read_csv(data_path, **read_csv_kwargs)
        except FileNotFoundError:
            print(f"Warning: Data path {data_path} does not exist.")
            return

        if 'bjd' in df.columns.tolist():
            df = df.rename(columns={'time':'date', 'bjd':'time'})

        assert set(['time', 'mnvel','errvel', 'tel']).issubset(df.columns), "Data must have columns: 'time', 'mnvel', 'errvel', 'tel'."
        
        data = pd.DataFrame()

        bintime, binmnvel, binerrvel, bintel = bintels(df.time, df.mnvel, df.errvel, df.tel, binsize=0.1)
        data['time'] = bintime
        data['mnvel'] = binmnvel
        data['errvel'] = binerrvel
        data['tel'] = bintel
        self.data = data

        if self.obs_start is None:
            self.obs_start = np.min(self.data.time) - 1
        if self.obs_end is None:
            self.obs_end = np.max(self.data.time) + 1

    def resample_data(self) -> pd.DataFrame:
        '''
        Resample the data according to the specified number of observations, cadence, and start/end dates.
        '''
        good_inds = []
        time_range_mask = self.data['time'] > self.obs_start
        time_range_mask &= self.data['time'] < self.obs_end
        inds = self.data[time_range_mask].index
        good_inds.append(inds[0])
        prev_time = self.data.loc[good_inds[0], 'time']

        for i in range(1, len(inds)):
            if len(good_inds) >= self.nobs:
                break
            ind = inds[i]
            if (self.data.loc[ind, 'time'] - prev_time) >= self.min_obs_cadence:
                good_inds.append(ind)
                prev_time = self.data.loc[ind, 'time']
            else:
                continue

        self.df_resampled = self.data.iloc[good_inds]
        return self.df_resampled