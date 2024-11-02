import radvel
import numpy as np
import pandas as pd

starname = "HIP 8152"
nplanets = 1
fitting_basis = "per tc secosw sesinw k"
bjd0 = 2457000
planet_letters = {1: "b"}

# From fit to the full data set using the base_config file.
k1_truth = 2.46
k1_err = 0.50

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(
    nplanets, basis="per tc e w k", planet_letters=planet_letters
)  # initialize Parameters object
# From MacDougall et al. (2023)
anybasis_params["per1"] = radvel.Parameter(value=10.751013)
anybasis_params["tc1"] = radvel.Parameter(value=1393.0862 + 2457000)
anybasis_params["e1"] = radvel.Parameter(value=0)
anybasis_params["w1"] = radvel.Parameter(value=0.000000)
anybasis_params["k1"] = radvel.Parameter(value=k1_truth)

data = pd.read_csv("hip8152/hip8152_rv.csv", comment="#")
data = data.rename(columns={"time": "date", "bjd": "time"})

time_base = np.median(data.time)
anybasis_params["dvdt"] = radvel.Parameter(value=0.0)
anybasis_params["curv"] = radvel.Parameter(value=0.0)

instnames = ["hires_j"]
ntels = len(instnames)
anybasis_params["gamma_hires_j"] = radvel.Parameter(value=0.0)
anybasis_params["jit_hires_j"] = radvel.Parameter(value=1.0)

params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)
mod = radvel.RVModel(params, time_base=time_base)

mod.params["per1"].vary = True
mod.params["tc1"].vary = True
mod.params["secosw1"].vary = False
mod.params["sesinw1"].vary = False

mod.params["dvdt"].vary = False
mod.params["curv"].vary = False

mod.params["gamma_hires_j"].vary = True
mod.params["jit_hires_j"].vary = True

priors = [
    radvel.prior.Gaussian(
        "per1", params["per1"].value, 7.1e-5
    ),  # From MacDougall et al. (2023)
    radvel.prior.Gaussian(
        "tc1", params["tc1"].value, 0.0034
    ),  # From MacDougall et al. (2023)
    radvel.prior.HardBounds("jit_hires_j", 0.0, 20.0),
]

# From MacDougall et al. (2023)
stellar = dict(mstar=0.94, mstar_err=0.03, rstar=0.96, rstar_err=0.02)
planet = dict(rp1=2.54, rp_err1=0.15)
