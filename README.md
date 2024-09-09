# Simulate RV observations by either resampling real data or creating synthetic data

There are some quirks in the package versions required to run the resampling experiments 
of real data and the injection-recovery experiments on the synthetic data. So in 
order recreate the analysis in the paper, you will probably need two different virtual 
environments. (Though if you'd rather just install and use the `rv_obs_sim` package for 
your own purposes, you only need the dependencies in the `basic_requirements.txt` 
file.)

### Resampling real data

An example jupyter notebook to recreate the resampling analysis can be found in `resampling/`.

`conda create -n resampling_env python=3.11.5`

`conda activate resampling_env`

`pip install cython==3.0.5`

`pip install -r resampling_requirements.txt`

`pip install .`

### Synthetic data

The `synthetic/` directory includes the scripts required to recreate the synthetic 
injection-recovery experiments. These require the `rvsearch` package, which requires 
an older version of `radvel`. 

`conda create -n synthetic_env python=3.11.5`

`conda activate synthetic_env`

`pip install cython==3.0.5`

`pip install -r synthetic_requirements.txt`

`pip install .`
