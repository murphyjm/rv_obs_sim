# Simulate RV observations by either resampling real data or creating synthetic data

There are some quirks in the package dependencies for the resampling experiments
of real data and the injection-recovery experiments on the synthetic data. `rvsearch`
(used to conduct the injection-recovery tests) requires an older version of `radvel`
than should be used for the resampling experiments. In order recreate the analysis in
the paper (Akana Murphy et al. 2024), you will probably need two different virtual
environments.

### Resampling real data

An example jupyter notebook to recreate the resampling analysis can be found
in `resampling_example/`.

`conda create -n resampling_env python=3.11.5`

`conda activate resampling_env`

`pip install cython==3.0.5`

`pip install -r resampling_requirements.txt`

`pip install .`

### Synthetic data

The `injection_recovery_example/` directory includes the scripts required to recreate the synthetic
injection-recovery experiments.

`conda create -n synthetic_env python=3.11.5`

`conda activate synthetic_env`

`pip install cython==3.0.5`

`pip install git+https://github.com/California-Planet-Search/rvsearch`

`pip install numpy==1.24.4`

`pip install .`

Note that on line `272` in `injection_recovery_example/run_simulation.py`, the object
`rvsearch.inject.Injections` accepts a `seed` keyword argument at instantiation. This is
not part of the standard `rvsearch` install, but a local change I made to enable consistency
between experiments. To add this functionality to your local `rvsearch` installation, update
the definition of the `Injections` class at `rvsearch.inject.Injections` accordingly:

```
class Injections(object):
    """
    ... docstring ...
    """

    def __init__(self, searchpath, plim, klim, elim, num_sim=1, full_grid=True, verbose=True, beta_e=False, seed=42):
        self.searchpath = searchpath
        self.plim = plim
        self.klim = klim
        self.elim = elim
        self.num_sim = num_sim
        self.full_grid = full_grid
        self.verbose = verbose
        self.beta_e = beta_e

        self.search = pickle.load(open(searchpath, 'rb'))
        self.search.verbose = False

        # CHANGES MADE BELOW
        # seed = np.round(self.search.data['time'].values[0] * 1000).astype(int)
        self.injected_planets = self.random_planets(seed)
        # CHANGES MADE ABOVE

        self.recoveries = self.injected_planets

        self.outdir = os.path.dirname(searchpath)
```
