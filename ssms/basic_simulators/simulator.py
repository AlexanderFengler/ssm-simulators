from . import boundary_functions as bf
from . import drift_functions as df
from ssms.config.config import model_config
import numpy as np
import pandas as pd
from copy import deepcopy
import cssm

"""
This module defines the basic simulator function which is the main
workshorse of the package.
In addition some utility functions are provided that help
with preprocessing the output of the simulator function.
"""


def _make_valid_dict(dict_in):
    """Turn all values in dictionary into numpy arrays and make sure,
    that all thetas are either scalars or vectors of the same length

    Arguments:
    ----------
        dict_in: dictionary
            Dictionary of parameters, potentially with different length / types per
            parameter (key)

    Returns:
    --------
        dict_in: dictionary
            Aligned to same size np.float32 np.arrays for every parameter
    """

    collect_lengths = []
    for key, value in dict_in.items():
        # Turn all values into numpy arrays
        if isinstance(value, list):
            dict_in[key] = np.array(value).astype(np.float32)
        elif isinstance(value, int) or isinstance(value, float):
            dict_in[key] = np.array([value]).astype(np.float32)

        # Squeeze all values to make sure they are 1d arrays
        dict_in[key] = np.squeeze(dict_in[key]).astype(np.float32)

        # Check if all thetas are either scalars or vectors of the same length
        if dict_in[key].ndim > 1:
            raise ValueError("Dimension of {} is greater than 1".format(key))
        elif dict_in[key].ndim > 0:
            collect_lengths.append(
                dict_in[key].shape[0]
            )  # add vector parameters to list

    if len(set(collect_lengths)) > 1:
        raise ValueError(
            "thetas have to be either scalars or same length for "
            "all thetas which are not scalars"
        )

    # If there were any thetas provided as vectors (and they had the same length),
    # tile all scalar thetas to that length
    if len(set(collect_lengths)) > 0:
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.tile(value, collect_lengths[0])
    else:  # Expand scalars to 1d arrays
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.expand_dims(value, axis=0)
    return dict_in


def _theta_dict_to_array(theta=dict(), model_param_list=None):
    """Converts theta dictionary to numpy array for use with simulator function"""
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")

    return np.stack([theta[param] for param in model_param_list], axis=1).astype(
        np.float32
    )


def _theta_array_to_dict(theta=None, model_param_list=None):
    """Converts theta array to dictionary for use with simulator function"""
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")
    elif theta is None:
        raise ValueError("theta array is not supplied")
    elif theta.ndim == 1 and len(model_param_list) != theta.shape[0]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    elif theta.ndim == 2 and len(model_param_list) != theta.shape[1]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    else:
        if theta.ndim == 1:
            theta = np.expand_dims(theta, axis=0)
        return {param: theta[:, i] for i, param in enumerate(model_param_list)}


# Basic simulators and basic preprocessing
def bin_simulator_output_pointwise(
    out=[0, 0],
    bin_dt=0.04,
    nbins=0,
):  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out: tuple
            Output of the 'simulator' function
        bin_dt: float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins: int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the
            number of bins automatically.

    Returns
    -------
        2d array. The first columns collects bin-identifiers
        by trial, the second column lists the corresponding choices.
    """
    out_copy = deepcopy(out)

    # Generate bins
    if nbins == 0:
        nbins = int(out["metadata"]["max_t"] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf

    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp

    out_copy[1][out_copy[1] == -1] = 0

    return np.concatenate([out_copy[0], out_copy[1]], axis=-1).astype(np.int32)


def bin_simulator_output(
    out=None,
    bin_dt=0.04,
    nbins=0,
    max_t=-1,
    freq_cnt=False,
):  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out : tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        max_t : int <default=-1>
            Override the 'max_t' metadata as part of the simulator output.
            Sometimes useful, but usually default will do the job.
        freq_cnt : bool <default=False>
            Decide whether to return proportions (default) or counts in bins.

    Returns
    -------
        A histogram of counts or proportions.

    """

    if max_t == -1:
        max_t = out["metadata"]["max_t"]

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros((nbins, len(out["metadata"]["possible_choices"])))

    for choice in out["metadata"]["possible_choices"]:
        counts[:, cnt] = np.histogram(out["rts"][out["choices"] == choice], bins=bins)[
            0
        ]
        cnt += 1

    if freq_cnt is False:
        counts = counts / out["metadata"]["n_samples"]

    return counts


def bin_arbitrary_fptd(
    out=None,
    bin_dt=0.04,
    nbins=256,
    nchoices=2,
    choice_codes=[-1.0, 1.0],
    max_t=10.0,
):  # ['v', 'a', 'w', 't', 'angle']
    """Takes in simulator output and returns a histogram of bin counts
    Arguments
    ---------
        out: tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size
            which in turn automatically determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        nchoices: int <default=2>
            Number of choices allowed by the simulator.
        choice_codes = list <default=[-1.0, 1.0]
            Choice labels to be used.
        max_t: float
            Maximum RT to consider.

    Returns
    -------
        2d array (nbins, nchoices): A histogram of bin counts
    """

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros((nbins, nchoices))

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins=bins)[0]
        cnt += 1
    return counts


def simulator(
    theta,
    model="angle",
    n_samples=1000,
    delta_t=0.001,
    max_t=20,
    no_noise=False,
    bin_dim=None,
    bin_pointwise=False,
    smooth_unif=True,
    return_option="full",
    random_state=None,
):
    """Basic data simulator for the models included in HDDM.

    Arguments
    ---------
        theta : list, numpy.array, dict or pd.DataFrame
            Parameters of the simulator. If 2d array, each row is treated as a 'trial'
            and the function runs n_sample * n_trials simulations.
        model: str <default='angle'>
            Determines the model that will be simulated.
        n_samples: int <default=1000>
            Number of simulation runs for each row in the theta argument.
        delta_t: float
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float
            Maximum reaction the simulator can reach
        no_noise: bool <default=False>
            Turn noise of (useful for plotting purposes mostly)
        bin_dim: int <default=None>
            Number of bins to use (in case the simulator output is
            supposed to come out as a count histogram)
        bin_pointwise: bool <default=False>
            Wheter or not to bin the output data pointwise.
            If true the 'RT' part of the data is now specifies the
            'bin-number' of a given trial instead of the 'RT' directly.
            You need to specify bin_dim as some number for this to work.
        return_option: str <default='full'>
            Determines what the function returns. Can be either
            'full' or 'minimal'. If 'full' the function returns
            a dictionary with keys 'rts', 'responses' and 'metadata', and
            metadata contains the model parameters and some additional
            information. 'metadata' is a simpler dictionary with less information
            if 'minimal' is chosen.
        random_state: int <default=None>
            Integer passed to random_seed function in the simulator.
            Can be used for reproducibility.

    Return
    ------
    dictionary where keys
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """
    # Preprocess theta to be a 2d numpy array with correct column ordering
    # (if supplied as 2d array or list in the first place,
    # user has to supply the correct ordering to begin with)

    if isinstance(theta, list):
        theta = np.asarray(theta).astype(np.float32)
    elif isinstance(theta, np.ndarray):
        theta = theta.astype(np.float32)
    elif isinstance(theta, dict):
        theta = _make_valid_dict(deepcopy(theta))
    elif isinstance(theta, pd.DataFrame):
        theta = theta.to_dict("list")
    else:
        try:
            import torch

            if isinstance(theta, torch.Tensor):
                theta = theta.numpy().astype(np.float32)
            else:
                raise ValueError(
                    "theta is not supplied as list, numpy array,"
                    " dictionary or torch tensor!"
                )
        except ImportError as e:
            raise e

    # Turn theta into array if it is a dictionary
    if isinstance(theta, dict):
        theta = _theta_dict_to_array(theta, model_config[model]["params"])

    # Adjust theta to be 2d array
    if len(theta.shape) < 2:
        theta = np.expand_dims(theta, axis=0)

    # Set number of trials to pass to simulator
    # based on shape of theta
    n_trials = theta.shape[0]

    # Make sure theta is np.float32
    theta = theta.astype(np.float32)

    # 2 choice models
    if no_noise:
        s = 0.0
    else:
        s = 1.0

    if model == "glob":
        x = cssm.glob_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alphar=theta[:, 3],
            g=theta[:, 4],
            t=theta[:, 5],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "test":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_deadline":
        x = cssm.ddm_flexbound_deadline(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            deadline=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # AF-TODO: Check what the intended purpose of 'ddm_legacy' was!
    if model == "ddm_hddm_base" or model == "ddm_legacy":
        x = cssm.ddm(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "angle":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 4]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "weibull_cdf" or model == "weibull":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 4], "beta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "levy":
        x = cssm.levy_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alpha_diff=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "levy_angle":
        x = cssm.levy_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alpha_diff=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "full_ddm" or model == "full_ddm2":
        x = cssm.full_ddm(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sz=theta[:, 4],
            sv=theta[:, 5],
            st=theta[:, 6],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "full_ddm_legacy" or model == "full_ddm_hddm_base":
        x = cssm.full_ddm_hddm_base(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sz=theta[:, 4],
            sv=theta[:, 5],
            st=theta[:, 6],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_sdv":
        x = cssm.ddm_sdv(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sv=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ornstein" or model == "ornstein_uhlenbeck":
        x = cssm.ornstein_uhlenbeck(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            g=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ornstein_angle":
        x = cssm.ornstein_uhlenbeck(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            g=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "gamma_drift":
        x = cssm.ddm_flex(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.constant,
            drift_fun=df.gamma_drift,
            boundary_multiplicative=True,
            boundary_params={},
            drift_params={"shape": theta[:, 4], "scale": theta[:, 5], "c": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "gamma_drift_angle":
        x = cssm.ddm_flex(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.angle,
            drift_fun=df.gamma_drift,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 4]},
            drift_params={"shape": theta[:, 5], "scale": theta[:, 6], "c": theta[:, 7]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ds_conflict_drift":
        x = cssm.ddm_flex(
            v=np.tile(np.array([0], dtype=np.float32), n_trials),
            a=theta[:, 0],
            z=theta[:, 1],
            t=theta[:, 2],
            s=s,
            boundary_fun=bf.constant,
            drift_fun=df.ds_conflict_drift,
            boundary_params={},
            drift_params={
                "init_p_t": theta[:, 3],
                "init_p_d": theta[:, 4],
                "slope_t": theta[:, 5],
                "slope_d": theta[:, 6],
                "fixed_p_t": theta[:, 7],
                "coherence_t": theta[:, 8],
                "coherence_d": theta[:, 9],
            },
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ds_conflict_drift_angle":
        x = cssm.ddm_flex(
            v=np.tile(np.array([0], dtype=np.float32), n_trials),
            a=theta[:, 0],
            z=theta[:, 1],
            t=theta[:, 2],
            s=s,
            boundary_fun=bf.angle,
            drift_fun=df.ds_conflict_drift,
            boundary_params={"theta": theta[:, 10]},
            boundary_multiplicative=False,
            drift_params={
                "init_p_t": theta[:, 3],
                "init_p_d": theta[:, 4],
                "slope_t": theta[:, 5],
                "slope_d": theta[:, 6],
                "fixed_p_t": theta[:, 7],
                "coherence_t": theta[:, 8],
                "coherence_d": theta[:, 9],
            },
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # Multi-particle models

    # 2 Choice
    if no_noise:
        s = np.tile(
            np.array(
                [
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            (n_trials, 1),
        )
    else:
        s = np.tile(
            np.array(
                [
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            (n_trials, 1),
        )

    if model == "race_2":
        x = cssm.race_model(
            v=theta[:, :2],
            a=theta[:, [2]],
            z=theta[:, 3:5],
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_2":
        x = cssm.race_model(
            v=theta[:, :2],
            a=theta[:, [2]],
            z=np.column_stack([theta[:, [3]], theta[:, [3]]]),
            t=theta[:, [4]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_2":
        x = cssm.race_model(
            v=theta[:, :2],
            a=theta[:, [2]],
            z=np.tile(
                np.array(
                    [
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                (n_trials, 1),
            ),
            t=theta[:, [3]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_angle_2":
        x = cssm.race_model(
            v=theta[:, :2],
            a=theta[:, [2]],
            z=np.column_stack([theta[:, [3]], theta[:, [3]]]),
            t=theta[:, [4]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_angle_2":
        x = cssm.race_model(
            v=theta[:, :2],
            a=theta[:, [2]],
            z=np.tile(
                np.array(
                    [
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                (n_trials, 1),
            ),
            t=theta[:, [3]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 4]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # 3 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (n_trials, 1))

    if model == "race_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=theta[:, 4:7],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            t=theta[:, [4]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_angle_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_angle_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            t=theta[:, [4]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=theta[:, 4:7],
            g=theta[:, [7]],
            b=theta[:, [8]],
            t=theta[:, [9]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_bias_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_z_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            g=theta[:, [4]],
            b=theta[:, [5]],
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_bias_angle_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 8]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_z_angle_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            g=theta[:, [4]],
            b=theta[:, [5]],
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 7]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # 4 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), (n_trials, 1))

    if model == "race_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=theta[:, 5:9],
            t=theta[:, [9]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_bias_angle_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 7]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "race_no_z_angle_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=theta[:, 5:9],
            g=theta[:, [9]],
            b=theta[:, [10]],
            t=theta[:, [11]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_bias_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            g=theta[:, [6]],
            b=theta[:, [7]],
            t=theta[:, [8]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_z_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_bias_angle_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            g=theta[:, [6]],
            b=theta[:, [7]],
            t=theta[:, [8]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 9]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "lca_no_z_angle_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1)),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 8]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # Seq / Parallel models (4 choice)
    if no_noise:
        s = 0.0
    else:
        s = 1.0

    z_vec = np.tile(np.array([0.5], dtype=np.float32), reps=n_trials)
    g_vec = np.tile(np.array([0.0], dtype=np.float32), reps=n_trials)
    g_vec_leak = np.tile(np.array([2.0], dtype=np.float32), reps=n_trials)
    a_zero_vec = np.tile(np.array([0.0], dtype=np.float32), reps=n_trials)
    s_pre_high_level_choice_zero_vec = np.tile(
        np.array([0.0], dtype=np.float32), reps=n_trials
    )
    s_pre_high_level_choice_one_vec = np.tile(
        np.array([1.0], dtype=np.float32), reps=n_trials
    )

    if model == "ddm_seq2":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            t=theta[:, 7],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_seq2_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_seq2_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 4],
                "theta": theta[:, 5],
                "scale": theta[:, 6],
                "alpha_gamma": theta[:, 7],
                "scale_gamma": theta[:, 8],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_seq2_angle_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_seq2_weibull_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 5], "beta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_par2":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            t=theta[:, 7],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_par2_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_par2_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 4],
                "theta": theta[:, 5],
                "scale": theta[:, 6],
                "alpha_gamma": theta[:, 7],
                "scale_gamma": theta[:, 8],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_par2_angle_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_par2_weibull_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 5], "beta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_adj":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            d=theta[:, 7],
            g=g_vec,
            t=theta[:, 8],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_adj_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_adj_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            g=g_vec,
            t=theta[:, 4],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_adj_angle_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_adj_weibull_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # ----- Ornstein version of mic2_adj ---------
    if model == "ddm_mic2_ornstein":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            d=theta[:, 7],
            g=theta[:, 8],
            t=theta[:, 9],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            g=theta[:, 4],
            t=theta[:, 5],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 6],
                "theta": theta[:, 7],
                "scale": theta[:, 8],
                "alpha_gamma": theta[:, 9],
                "scale_gamma": theta[:, 10],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            g=theta[:, 4],
            t=theta[:, 5],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 6],
                "theta": theta[:, 7],
                "scale": theta[:, 8],
                "alpha_gamma": theta[:, 9],
                "scale_gamma": theta[:, 10],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_angle_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_weibull_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 7], "beta": theta[:, 8]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=theta[:, 5],
            t=theta[:, 6],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 7], "beta": theta[:, 8]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # Leak version of mic2
    if model == "ddm_mic2_leak":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            d=theta[:, 7],
            g=g_vec_leak,
            t=theta[:, 8],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            g=g_vec_leak,
            t=theta[:, 4],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_angle_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_weibull_no_bias":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s_pre_high_level_choice=s_pre_high_level_choice_one_vec,
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_angle_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            g=g_vec_leak,
            t=theta[:, 5],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise":
        x = cssm.ddm_flexbound_mic2_ornstein(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            g=g_vec_leak,
            t=theta[:, 4],
            s=s,
            s_pre_high_level_choice=s_pre_high_level_choice_zero_vec,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # ----------------- High level dependent noise scaling --------------
    if model == "ddm_mic2_multinoise_no_bias":
        x = cssm.ddm_flexbound_mic2_multinoise(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "ddm_mic2_multinoise_angle_no_bias":
        x = cssm.ddm_flexbound_mic2_multinoise(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )
    if model == "ddm_mic2_multinoise_weibull_no_bias":
        x = cssm.ddm_flexbound_mic2_multinoise(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )
    if model == "ddm_mic2_multinoise_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_mic2_multinoise(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # ----------------- Tradeoff models -----------------
    if model == "tradeoff_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "tradeoff_angle_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "tradeoff_weibull_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    if model == "tradeoff_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
            return_option=return_option,
            smooth=smooth_unif,
        )

    # Output compatibility
    if n_trials == 1:
        x["rts"] = np.squeeze(x["rts"], axis=1)
        x["choices"] = np.squeeze(x["choices"], axis=1)
    if n_trials > 1 and n_samples == 1:
        x["rts"] = np.squeeze(x["rts"], axis=0)
        x["choices"] = np.squeeze(x["choices"], axis=0)

    x["metadata"]["model"] = model

    # Adjust in output to binning choice
    if bin_dim == 0 or bin_dim is None:
        return x
    elif bin_dim > 0 and n_trials == 1 and not bin_pointwise:
        binned_out = bin_simulator_output(x, nbins=bin_dim)
        return {"data": binned_out, "metadata": x["metadata"]}
    elif bin_dim > 0 and n_trials == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins=bin_dim)
        return {
            "rts": np.expand_dims(binned_out[:, 0], axis=1),
            "choices": np.expand_dims(binned_out[:, 1], axis=1),
            "metadata": x["metadata"],
        }
    elif bin_dim > 0 and n_trials > 1 and n_samples == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins=bin_dim)
        return {
            "rts": np.expand_dims(binned_out[:, 0], axis=1),
            "choices": np.expand_dims(binned_out[:, 1], axis=1),
            "metadata": x["metadata"],
        }
    elif bin_dim > 0 and n_trials > 1 and n_samples > 1 and bin_pointwise:
        return (
            "currently n_trials > 1 and n_samples > 1, "
            "will not work together with bin_pointwise"
        )
    elif bin_dim > 0 and n_trials > 1 and not bin_pointwise:
        return "currently binned outputs not implemented for multi-trial simulators"
    elif bin_dim == -1:
        return "invalid bin_dim"
