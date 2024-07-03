import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import radvel
from matplotlib import rcParams
from radvel.plot import orbit_plots
from tqdm import tqdm

rcParams["font.size"] = 14
import copy

from matplotlib import colors
from matplotlib.ticker import MultipleLocator


class SimGrid:

    def __init__(
        self,
        moc_grid,
        nrv_grid,
        base_config_file,
        fit_config_file,
        sys_name="",
        config_id="",
        data_file=None,
        obs_start_end=(None, None),
        read_csv_kwargs={},
        random_seed=42,
        tel="hires_j",
        time_jitter=0,
        astro_rv_jitter=0,
        tel_rv_jitter=0,
        errvel_scale=2,
        max_baseline=3650,
        binning=False,
        plot_title_name="",
    ) -> None:

        self.sys_name = sys_name
        if plot_title_name != "":
            self.plot_title_name = plot_title_name
        else:
            self.plot_title_name = sys_name
        self.config_id = config_id

        self.moc_grid = moc_grid
        self.nrv_grid = nrv_grid

        self.base_config_file = base_config_file
        base_config_file_obj, base_post = radvel.utils.initialize_posterior(
            self.base_config_file
        )
        self.base_config_file_obj = base_config_file_obj
        self.base_post = base_post

        self.fit_config_file = fit_config_file
        fit_config_file_obj, _ = radvel.utils.initialize_posterior(self.fit_config_file)
        self.fit_config_file_obj = fit_config_file_obj

        # Whether or not to bin the RV data before fitting
        self.binning = binning

        self.data_file = data_file
        self.obs_start, self.obs_end = obs_start_end
        if self.data_file is not None:
            self.__load_data(**read_csv_kwargs)

        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.tel = tel
        self.time_jitter = time_jitter
        self.astro_rv_jitter = astro_rv_jitter
        self.tel_rv_jitter = tel_rv_jitter
        self.errvel_scale = errvel_scale

        self.max_baseline = max_baseline

    def __load_data(self, **read_csv_kwargs):

        try:
            df = pd.read_csv(self.data_file, **read_csv_kwargs)
        except FileNotFoundError:
            print(f"Warning: Data path {self.data_file} does not exist.")
            return

        if "bjd" in df.columns.tolist():
            df = df.rename(columns={"time": "date", "bjd": "time"})

        assert set(["time", "mnvel", "errvel", "tel"]).issubset(
            df.columns
        ), "Data must have columns: 'time', 'mnvel', 'errvel', 'tel'."

        if self.binning:
            data = pd.DataFrame()
            bintime, binmnvel, binerrvel, bintel = radvel.utils.bintels(
                df.time.values,
                df.mnvel.values,
                df.errvel.values,
                df.tel.values,
                binsize=0.1,
            )
            data["time"] = bintime
            data["mnvel"] = binmnvel
            data["errvel"] = binerrvel
            data["tel"] = bintel
            self.data = data
        else:
            self.data = df

        if self.obs_start is None:
            self.obs_start = np.min(self.data.time) - 1
        if self.obs_end is None:
            self.obs_end = np.max(self.data.time) + 1

    def __set_parent_synth_data_grid(self):
        """
        Should only have to generate the synthetic RVs once, but on a large grid, and then you can just
        resample them later based on the number of RVs requested and the MOC.
        """

        time_grid = (
            np.arange(self.max_baseline) + self.base_config_file_obj.time_base
        ).astype(float)
        time_grid += np.random.normal(scale=self.time_jitter, size=len(time_grid))

        rv_tot = np.zeros(len(time_grid))

        for (
            pl_ind
        ) in (
            self.base_config_file_obj.planet_letters.keys()
        ):  # Need to input correct K-amplitudes in config file.

            # Extract orbit parameters
            p = self.base_post.params[f"per{pl_ind}"].value
            tc = self.base_post.params[f"tc{pl_ind}"].value
            if (
                "secosw" in self.base_config_file_obj.fitting_basis
                and "sesinw" in self.base_config_file_obj.fitting_basis
            ):
                secosw = self.base_post.params[f"secosw{pl_ind}"].value
                sesinw = self.base_post.params[f"secosw{pl_ind}"].value
                e = secosw**2 + sesinw**2
                omega = np.arctan(sesinw / secosw)
                if np.isnan(omega):
                    omega = 0.0
            else:
                e = self.base_post.params[f"e{pl_ind}"].value
                omega = self.base_post.params[f"omega{pl_ind}"].value
            tp = radvel.orbit.timetrans_to_timeperi(tc, p, e, omega)
            k = self.base_post.params[f"k{pl_ind}"].value

            orbel = (p, tp, e, omega, k)
            rv_tot += radvel.kepler.rv_drive(time_grid, orbel)

        # Add background trend if applicable. Need to input correct dvdt and curv values in config file.
        rv_tot += (
            time_grid - self.base_config_file_obj.time_base
        ) * self.base_post.params["dvdt"].value
        rv_tot += (
            time_grid - self.base_config_file_obj.time_base
        ) ** 2 * self.base_post.params["curv"].value

        # Add random noise
        rv_tot += np.random.normal(
            scale=np.sqrt(self.astro_rv_jitter**2 + self.tel_rv_jitter**2),
            size=len(rv_tot),
        )

        self.time_grid, self.mnvel_grid, self.errvel_grid = (
            time_grid,
            rv_tot,
            self.errvel_scale * np.ones(len(rv_tot)),
        )

    def get_ksim_grid(
        self, disable_progress_bar=False, save_posts=False, verbose=False
    ):
        """
        For all cells in the moc vs nrv grid, run a MAP fit to get recovered K-amplitude values
        """

        if self.data_file is None:
            self.__set_parent_synth_data_grid()

        fit_config_file_obj, fit_post = radvel.utils.initialize_posterior(
            self.fit_config_file
        )

        if save_posts:
            post_grid = np.empty((len(self.moc_grid), len(self.nrv_grid)), dtype=object)

        ksim_grid = np.empty(
            (len(self.moc_grid), len(self.nrv_grid), fit_post.model.num_planets)
        )

        # TODO: Clean up this big outer if else statement
        # SYNTHETIC DATA
        if self.data_file is None:
            i = 0
            time_grid_inds = np.arange(self.max_baseline)
            for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
                moc_mask = time_grid_inds % moc == 0

                for j, nrv in enumerate(self.nrv_grid):

                    # Note: fit_config_file_obj.time_base and self.base_config_file_obj.time_base should probably be the same.
                    mask = moc_mask & (time_grid_inds / moc < nrv)

                    fit_mod = radvel.RVModel(
                        fit_post.params, time_base=fit_config_file_obj.time_base
                    )
                    fit_like = None
                    # TODO: Implement Composite likelihood objects so that these fits can be done with data from multiple instruments.
                    if (
                        hasattr(fit_config_file_obj, "hnames")
                        and len(fit_config_file_obj.hnames) > 0
                    ):  # If it is a GP-enabled config file, use a GP likelihood
                        tel = fit_config_file_obj.instnames[
                            0
                        ]  # HACK for single telescope data sets right now
                        fit_like = radvel.likelihood.GPLikelihood(
                            fit_mod,
                            self.time_grid[mask],
                            self.mnvel_grid[mask],
                            self.errvel_grid[mask],
                            hnames=fit_config_file_obj.hnames[tel],
                        )

                    else:
                        fit_like = radvel.likelihood.RVLikelihood(
                            fit_mod,
                            self.time_grid[mask],
                            self.mnvel_grid[mask],
                            self.errvel_grid[mask],
                        )
                    post = radvel.posterior.Posterior(fit_like)
                    post.priors += self.fit_config_file_obj.priors
                    post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
                    planet_letter_keys = list(fit_config_file_obj.planet_letters.keys())
                    k_maps = np.array(
                        [post.params[f"k{i}"].value for i in planet_letter_keys]
                    )

                    for k in range(fit_post.model.num_planets):
                        ksim_grid[i, j, k] = k_maps[k]

                    if save_posts:
                        post_grid[i, j] = post

                i += 1
        # REAL DATA
        else:
            time_range_mask = self.data["time"] > self.obs_start
            time_range_mask &= self.data["time"] < self.obs_end
            inds = self.data[time_range_mask].index

            i = 0
            for moc in tqdm(self.moc_grid, disable=disable_progress_bar):
                for j, nrv in enumerate(self.nrv_grid):
                    good_inds = [inds[0]]
                    prev_time = self.data.loc[good_inds[0], "time"]
                    for k in range(1, len(inds)):
                        if len(good_inds) >= nrv:
                            break
                        ind = inds[k]
                        if (self.data.loc[ind, "time"] - prev_time) >= moc:
                            good_inds.append(ind)
                            prev_time = self.data.loc[ind, "time"]
                        else:
                            continue

                    resampled_data = self.data.iloc[good_inds].reset_index(drop=True)
                    resampled_fit_post_params = copy.deepcopy(fit_post.params)
                    for param in fit_post.params.keys():
                        if "gamma" in param or "jit" in param:
                            if param.split("_")[-1] not in resampled_data.tel.unique():
                                resampled_fit_post_params.pop(param)

                    fit_mod = radvel.RVModel(
                        resampled_fit_post_params,
                        time_base=fit_config_file_obj.time_base,
                    )
                    fit_like = None

                    if (
                        hasattr(fit_config_file_obj, "hnames")
                        and len(fit_config_file_obj.hnames) > 0
                    ):  # If it is a GP-enabled config file, use a GP likelihood
                        tel = fit_config_file_obj.instnames[
                            0
                        ]  # HACK for single telescope data sets right now
                        fit_like = radvel.likelihood.GPLikelihood(
                            fit_mod,
                            resampled_data.time,
                            resampled_data.mnvel,
                            resampled_data.errvel,
                            hnames=fit_config_file_obj.hnames[tel],
                        )
                    else:
                        if len(resampled_data.tel.unique()) > 1:
                            # Implementing functionality to fit with multiple instruments
                            # initialize Likelihood objects for each instrument
                            iparams = radvel.basis._copy_params(
                                resampled_fit_post_params
                            )
                            telgrps = resampled_data.groupby("tel").groups
                            likes = {}

                            for inst in telgrps.keys():
                                likes[inst] = radvel.likelihood.RVLikelihood(
                                    fit_mod,
                                    resampled_data.iloc[telgrps[inst]].time,
                                    resampled_data.iloc[telgrps[inst]].mnvel,
                                    resampled_data.iloc[telgrps[inst]].errvel,
                                    suffix="_" + inst,
                                )
                                likes[inst].params["gamma_" + inst] = iparams[
                                    "gamma_" + inst
                                ]
                                likes[inst].params["jit_" + inst] = iparams[
                                    "jit_" + inst
                                ]

                            fit_like = radvel.likelihood.CompositeLikelihood(
                                list(likes.values())
                            )
                        else:
                            fit_like = radvel.likelihood.RVLikelihood(
                                fit_mod,
                                resampled_data.time,
                                resampled_data.mnvel,
                                resampled_data.errvel,
                            )

                    post = radvel.posterior.Posterior(fit_like)
                    resampled_fit_priors = []
                    for prior in self.fit_config_file_obj.priors:
                        try:
                            param_name = prior.param
                            if "gamma" in param_name or "jit" in param_name:
                                if (
                                    param_name.split("_")[-1]
                                    not in resampled_data.tel.unique()
                                ):
                                    continue
                                else:
                                    resampled_fit_priors.append(prior)
                            else:
                                resampled_fit_priors.append(prior)
                        except AttributeError:
                            resampled_fit_priors.append(prior)
                            pass

                    post.priors += resampled_fit_priors
                    post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
                    planet_letter_keys = list(fit_config_file_obj.planet_letters.keys())
                    k_maps = np.array(
                        [post.params[f"k{i}"].value for i in planet_letter_keys]
                    )

                    for k in range(fit_post.model.num_planets):
                        ksim_grid[i, j, k] = k_maps[k]

                    if save_posts:
                        post_grid[i, j] = post

                i += 1

        self.ksim_grid = ksim_grid
        if save_posts:
            self.post_grid = post_grid

        return ksim_grid

    def get_ksim_over_ktruth(self):
        """
        Convenience function.
        """
        ksim_over_ktruth = np.copy(self.ksim_grid)
        for k in self.fit_config_file_obj.planet_letters.keys():
            ksim_over_ktruth[:, :, k - 1] /= self.base_post.params[f"k{k}"].value

        self.ksim_over_ktruth = ksim_over_ktruth
        return ksim_over_ktruth

    def get_ksim_minus_ktruth_over_kerr(self):
        """
        Convenience function.
        """
        ksim_minus_ktruth_over_kerr = np.copy(self.ksim_over_ktruth)
        for pl_ind, pl_letter in self.fit_config_file_obj.planet_letters.items():
            ksim_minus_ktruth_over_kerr[:, :, pl_ind - 1] *= self.base_post.params[
                f"k{pl_ind}"
            ].value
            ksim_minus_ktruth_over_kerr[:, :, pl_ind - 1] -= self.base_post.params[
                f"k{pl_ind}"
            ].value
            ksim_minus_ktruth_over_kerr[:, :, pl_ind - 1] /= getattr(
                self.base_config_file_obj, f"k{pl_ind}_err"
            )
        self.ksim_minus_ktruth_over_kerr = ksim_minus_ktruth_over_kerr

        return ksim_minus_ktruth_over_kerr

    def convert_moc_to_grid_ind(self, moc):
        moc_ind = np.argwhere(self.moc_grid == moc)
        assert len(moc_ind) > 0, f"MOC = {moc} days is not in the simulation grid."
        moc_ind = moc_ind.flatten()[0]
        return moc_ind

    def convert_nrv_to_grid_ind(self, nrv):
        nrv_ind = np.argwhere(self.nrv_grid == nrv)
        assert len(nrv_ind) > 0, f"N_RV = {nrv} is not in the simulation grid."
        nrv_ind = nrv_ind.flatten()[0]
        return nrv_ind

    def plot_grid_cell_fit(self, moc, nrv, savefname=None):
        """
        Given an observing cadence and a number of RVs, plot the best-fit solution the results in the recovered K-amplitude.

        08/17/23: Some weird bug where you have to run this function twice to make it update correctly.

        TODO: Fix plotting bug.
        """
        assert hasattr(
            self, "post_grid"
        ), "SimGrid object does not have a post_grid attribute. Try re-running get_ksim_over_ktruth_grid() with save_posts=True."

        moc_ind = self.convert_moc_to_grid_ind(moc)
        nrv_ind = self.convert_nrv_to_grid_ind(nrv)

        rvplot = orbit_plots.MultipanelPlot(
            self.post_grid[moc_ind, nrv_ind], legend=False
        )
        fig, axes = rvplot.plot_multipanel()
        axes[0].set_title(
            f"MOC = {moc} d; $N_\mathrm{{RV}} = {nrv}$; Added RV jitter = {np.sqrt(self.tel_rv_jitter**2 + self.astro_rv_jitter**2):.1f} m/s"
        )
        if savefname is not None:
            fig.savefig(savefname, bbox_inches="tight")
        return fig, axes

    def make_grid_plot(
        self, grid, cbar_units="ratio", savefig=False, save_ext=".pdf", save_dpi=600
    ):
        """
        Make the nice plot of the simulation grid results.
        """
        cbar_units = cbar_units.lower()
        allowed_cbar_units = ["ratio", "diff_over_sigma"]
        assert (
            cbar_units in allowed_cbar_units
        ), f"'cbar_units' must be one of {allowed_cbar_units}"

        cmap = plt.cm.bwr  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

        # Parameters of the colorbar
        if cbar_units == "diff_over_sigma":
            vmin = -5
            vmax = 5
            cbar_xticks = np.linspace(vmin, vmax, (vmax - vmin) + 1)
            # cbar_xticks = np.sort(np.append(cbar_xticks, [-0.5, 0.5]))
            bounds = cbar_xticks[cbar_xticks != 0]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            cbar_xtick_labels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

        elif cbar_units == "ratio":
            vmin = 0
            vmax = 2
            cbar_xticks = np.linspace(vmin, vmax, (vmax - vmin) * 4 + 1)
            # cbar_xticks = np.sort(np.append(cbar_xticks, [0.90, 1.10]))
            bounds = cbar_xticks[cbar_xticks != 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            cbar_xtick_labels = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75, 2]

        nplanets = len(self.fit_config_file_obj.planet_letters.items())
        figsize = (10, 6 * nplanets)
        fig, axes = plt.subplots(ncols=1, nrows=nplanets, figsize=figsize, sharex=True)
        axis_ind = 0
        for pl_ind, pl_letter in self.fit_config_file_obj.planet_letters.items():

            if nplanets == 1:
                ax = axes
            else:
                ax = axes[axis_ind]
            px = (self.nrv_grid[-1] - self.nrv_grid[0]) / len(self.nrv_grid)
            py = (self.moc_grid[-1] - self.moc_grid[0]) / len(self.moc_grid)
            extent = [
                self.nrv_grid[0] - px / 2,
                self.nrv_grid[-1] + px / 2,
                self.moc_grid[0] - py / 2,
                self.moc_grid[-1] + py / 2,
            ]
            im = ax.imshow(
                grid[:, :, pl_ind - 1],
                cmap="bwr",
                origin="lower",
                extent=extent,
                aspect="auto",
                norm=norm,
            )

            if self.moc_grid[0] > 0:
                ax.set_yticks(np.append(self.moc_grid[0], self.moc_grid[4::5]))
                ax.set_yticks(self.moc_grid, minor=True)
            else:
                ax.set_yticks(np.append(self.moc_grid[1], self.moc_grid[5::5]))
                ax.set_yticks(self.moc_grid, minor=True)

            ax.set_xticks(self.nrv_grid[::5])
            ax.set_xticks(self.nrv_grid, minor=True)

            per_twin = self.base_post.params[f"per{pl_ind}"].value
            ax2 = ax.twinx()
            ax2.set_ylim(extent[2] / per_twin, extent[3] / per_twin)
            if per_twin > np.max(self.moc_grid):
                ax2.yaxis.set_major_locator(MultipleLocator(0.005))
                ax2.yaxis.set_minor_locator(MultipleLocator(0.001))
            elif per_twin < 1:
                ax2.yaxis.set_major_locator(MultipleLocator(5))
                ax2.yaxis.set_minor_locator(MultipleLocator(1))
            else:
                ax2.yaxis.set_major_locator(MultipleLocator(0.50))
                ax2.yaxis.set_minor_locator(MultipleLocator(0.125))
            ax2.set_ylabel(
                f"Obs. cadence / $P_\mathrm{{{pl_letter}}}$ ($P_\mathrm{{{pl_letter}}} = {per_twin:.2f}$ days)",
                rotation=270,
                labelpad=25,
            )

            if self.base_post.model.num_planets > 1:
                pl_hlines_ind = None
                per_mults = None
                if pl_ind == 1:
                    pl_hlines_ind = 2
                    if self.base_post.params["per2"].value > np.max(self.moc_grid):
                        per_mults = np.array([0.01, 0.025])
                    else:
                        per_mults = np.array([0.25, 0.5, 1])
                else:
                    pl_hlines_ind = pl_ind - 1
                    if self.base_post.params[f"per{pl_hlines_ind}"].value < 1:
                        per_mults = np.array([1, 10])
                    else:
                        per_mults = np.array([0.5, 1, 2])
                per_hlines = self.base_post.params[f"per{pl_hlines_ind}"].value
                pl_letter_hlines = self.base_config_file_obj.planet_letters[
                    pl_hlines_ind
                ]
                ax.hlines(
                    per_hlines * per_mults,
                    extent[0],
                    extent[1],
                    ls="--",
                    lw=3,
                    color="k",
                )
                for mult in per_mults:
                    annotate_str = None
                    if mult == 1:
                        annotate_str = (
                            f"$P_\mathrm{{{pl_letter_hlines}}} = {per_hlines:.2f}$ days"
                        )
                    else:
                        annotate_str = (
                            f"$P_\mathrm{{{pl_letter_hlines}}} \\times {mult}$"
                        )
                    ax.text(
                        extent[1] - 0.5,
                        per_hlines * mult + 0.15,
                        annotate_str,
                        ha="right",
                        va="bottom",
                        fontsize=16,
                    )

            xmin, xmax = ax.get_xlim()
            if self.data_file is not None:
                data_type = "Re-sampled real"
                ylabel = "Minimum obs. cadence [days]"

                time_range_mask = self.data["time"] > self.obs_start
                time_range_mask &= self.data["time"] < self.obs_end
                inds = self.data[time_range_mask].index
                nrv_max_for_moc_arr = np.empty(len(self.moc_grid))
                for i, moc in enumerate(self.moc_grid):
                    good_inds = [inds[0]]
                    prev_time = self.data.loc[good_inds[0], "time"]
                    for k in range(1, len(inds)):
                        ind = inds[k]
                        if (self.data.loc[ind, "time"] - prev_time) >= moc:
                            good_inds.append(ind)
                            prev_time = self.data.loc[ind, "time"]
                        else:
                            continue
                    nrv_max_for_moc_arr[i] = len(good_inds)
                    ax.fill_between(
                        np.arange(nrv_max_for_moc_arr[i], xmax + 1),
                        moc - py / 2,
                        moc + py / 2,
                        color="lightgray",
                        alpha=1,
                    )
                save_id = "real"

            else:
                data_type = "Synthetic"
                ylabel = "Obs. cadence [days]"
                save_id = "synth"
            ax.set_ylabel(ylabel)
            ax.set_xlim(xmin, xmax)

            # Make ticks go in or out
            ax.tick_params(
                axis="x", direction="out", which="both", top=True, bottom=True
            )
            fname = self.fit_config_file.split("/")[-1]
            model_str = fname.split("_")[1]
            if model_str == "base":
                model_str = "baseline"
            ax.text(
                0.5,
                0.99,
                f"{self.plot_title_name} {pl_letter}\nModel: {model_str}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=16,
            )

            axis_ind += 1

        if nplanets > 1:
            fig.subplots_adjust(hspace=0.05)
            cbar_ax = fig.add_axes([0.1, 0.03, 0.85, 0.025])
        else:
            cbar_ax = fig.add_axes([0.1, -0.05, 0.85, 0.05])
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation="horizontal",
            norm=norm,
            ticklocation="bottom",
            spacing="proportional",
            ticks=bounds,
            boundaries=bounds,
        )
        if cbar_units == "diff_over_sigma":
            cbar_label = f"$(K_\mathrm{{fit}} - K_\mathrm{{baseline}}) / \sigma_{{K_\mathrm{{baseline}}}}$"
        elif cbar_units == "ratio":
            cbar_label = f"$K_\mathrm{{fit}}/K_\mathrm{{baseline}}$"
        cbar.set_label(cbar_label)
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.set_ticks(cbar_xticks)
        cbar.ax.xaxis.set_ticklabels(cbar_xtick_labels, fontdict={"fontsize": 16})

        ax.set_xlabel("$N_\mathrm{rv}$")

        if savefig:
            fname = f"{self.sys_name}/{self.sys_name}_{save_id}_grid_{self.config_id}_config_{cbar_units}{save_ext}"
            save_fig_kwargs = {"bbox_inches": "tight"}
            if save_ext == ".png":
                save_fig_kwargs["dpi"] = save_dpi
                save_fig_kwargs["facecolor"] = "white"
            fig.savefig(fname, **save_fig_kwargs)

        return fig, [axes, cbar]
