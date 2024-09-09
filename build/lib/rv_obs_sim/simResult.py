class SimResult:
    '''
    Container object to store metadata of simulation result.
    '''
    
    def __init__(self, sim, post) -> None:
        
        self.sim = sim
        
        try:
            self.nobs = len(sim.df_resampled)
        except AttributeError:
            self.nobs_goal = sim.nobs
        
        self.nobs = sim.nobs
        self.min_obs_cadence = sim.min_obs_cadence
        
        self.post = post
        self.nplanets = post.model.num_planets
        
        self.kamp_map = {i:post.likelihood.params[f'k{i}'].value for i in range(1, self.nplanets + 1)}
        if hasattr(post, 'medparams'):
            self.kamp_med = {i:post.medparams[f'k{i}'] for i in range(1, self.nplanets + 1)}
            self.sigma_kamp = {i:post.uparams[f'k{i}'] for i in range(1, self.nplanets + 1)}